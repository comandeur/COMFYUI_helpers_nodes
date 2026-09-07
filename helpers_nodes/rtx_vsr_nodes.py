"""NVIDIA RTX Video Super Resolution, driven from inside the graph.

ComfyUI-RTX-Video-Suite wraps the same SDK but works on files: it reads a video
off disk and writes an upscaled one back. That means leaving the workflow,
re-encoding, and a round trip through ffmpeg every time.

Two upscale nodes share one upscale loop and one cached inference engine:

* ``RTXVideoUpscale`` -- IMAGE in, IMAGE out, AUDIO carried through untouched.
  Stays in the graph, at the cost of the upscaled batch existing as one tensor.
* ``RTXVideoUpscaleToFile`` -- IMAGE in, video file out. Each frame goes to the
  encoder and is dropped, so a long upscale never has to fit in RAM.

Frames reach the SDK on the GPU over DLPack, one at a time, so VRAM cost is one
input frame plus one output frame regardless of how long the batch is.

Requires the NVIDIA Video Effects (VFX) SDK, i.e. an importable ``nvvfx``, and an
RTX GPU. Without it the nodes still register and only fail when executed.
Additional same-resolution nodes provide artifact cleanup, human segmentation
and background blur. The latter two use the native SDK via rtx_vfx_native.
"""
import os
import threading

import torch

import folder_paths
import comfy.model_management as model_management
from comfy.utils import ProgressBar

# Fallback list so the combo widget still has its options when the SDK is
# missing (the node registers, and only errors out when actually executed).
QUALITY_LEVELS = [
    "BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA",
    "DENOISE_LOW", "DENOISE_MEDIUM", "DENOISE_HIGH", "DENOISE_ULTRA",
    "DEBLUR_LOW", "DEBLUR_MEDIUM", "DEBLUR_HIGH", "DEBLUR_ULTRA",
    "HIGHBITRATE_LOW", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_HIGH", "HIGHBITRATE_ULTRA",
]

# Modes that enhance in place instead of upscaling: the SDK wants output dims
# equal to input dims for those.
SAME_SIZE_PREFIXES = ("DENOISE_", "DEBLUR_")

# The SDK itself is happy with any output size (odd numbers included), but video
# encoders downstream are not: h264/h265 need even dimensions, and a latent
# round trip wants multiples of 8.
ALIGN_OPTIONS = {"exact": 1, "even": 2, "multiple of 8": 8}

# float32 is what ComfyUI hands around as IMAGE; float16 halves the RAM the
# output batch needs and every node worth its salt accepts it.
OUTPUT_DTYPES = {"float32": torch.float32, "float16": torch.float16}

# kept in sync with video_encode.CONTAINER_FORMATS, but spelled out here so the
# widget still has its options when PyAV is missing
VIDEO_FORMATS = ["mp4", "avi"]


def get_nvvfx():
    """Import the SDK, or raise something a user can act on."""
    try:
        import nvvfx
    except ImportError as e:
        raise RuntimeError(
            "NVIDIA Video Effects SDK not available: 'import nvvfx' failed.\n"
            "Install the VFX SDK (NVIDIA AI for Media, formerly Maxine) into the "
            "python environment ComfyUI runs on, then restart ComfyUI."
        ) from e
    return nvvfx


def get_quality_enum():
    nvvfx = get_nvvfx()
    try:
        from nvvfx.effects.video_super_res import QualityLevel
    except ImportError:
        # older/other SDK layouts nested it on the effect class
        QualityLevel = getattr(nvvfx.VideoSuperRes, "QualityLevel", None)
        if QualityLevel is None:
            raise RuntimeError("Could not locate QualityLevel in the installed nvvfx.")
    return QualityLevel


def get_quality_options():
    """Widget options: the enum from the installed SDK, or the static fallback."""
    try:
        return [q.name for q in get_quality_enum()]
    except Exception:
        return list(QUALITY_LEVELS)


class _VSRSession:
    """One VideoSuperRes effect kept alive between executions.

    ``load()`` builds/deserialises the inference engine, which is far too slow to
    redo per queued prompt. Quality and output size can both be changed on a
    loaded effect, so a single cached instance covers every setting.
    """

    def __init__(self):
        self.effect = None
        self.device = None

    def get(self, quality, out_width, out_height, device_index):
        nvvfx = get_nvvfx()
        QualityLevel = get_quality_enum()
        level = QualityLevel[quality]

        if self.effect is not None and self.device != device_index:
            self.close()
        if self.effect is None:
            self.effect = nvvfx.VideoSuperRes(level, device=device_index)
            self.device = device_index

        effect = self.effect
        effect.quality = level
        effect.output_width = out_width
        effect.output_height = out_height
        if not effect.is_loaded or effect.needs_reload:
            effect.load()
        return effect

    def close(self):
        if self.effect is not None:
            try:
                self.effect.close()
            except Exception:
                pass
        self.effect = None
        self.device = None


_SESSION = _VSRSession()


def resolve_output_size(width, height, quality, scale, custom_width, custom_height,
                        align="even"):
    """Output size: explicit dims win over the multiplier, then alignment."""
    if quality.startswith(SAME_SIZE_PREFIXES):
        # denoise/deblur run at source resolution by definition
        return width, height

    if custom_width == 0 and custom_height == 0:
        out_w, out_h = width * scale, height * scale
    elif custom_height == 0:
        out_w = custom_width
        out_h = height * (custom_width / width)
    elif custom_width == 0:
        out_h = custom_height
        out_w = width * (custom_height / height)
    else:
        out_w, out_h = custom_width, custom_height

    step = ALIGN_OPTIONS.get(align, 1)
    out_w = max(step, round(out_w / step) * step)
    out_h = max(step, round(out_h / step) * step)
    return int(out_w), int(out_h)


def allocate_output(frame_count, out_height, out_width, dtype):
    """The upscaled batch, with an error that says what to do when it won't fit.

    An IMAGE output has to exist as one tensor -- that is the socket's contract,
    not a choice this node makes. What can be chosen is how to stay under it.
    """
    try:
        return torch.empty((frame_count, out_height, out_width, 3), dtype=dtype)
    except (RuntimeError, MemoryError) as e:
        gib = frame_count * out_height * out_width * 3 * dtype.itemsize / 2 ** 30
        raise RuntimeError(
            f"Could not allocate the output batch: {frame_count} frames of "
            f"{out_width}x{out_height} in {dtype} is {gib:.1f} GiB of RAM, in one "
            "contiguous block, and that is before the input batch already held "
            "alongside it.\n"
            "Any IMAGE-based path pays this, not just this node. Options:\n"
            "  - run the graph through a VHS Meta Batch Manager, so only "
            "frames_per_batch frames exist at a time (this node is stateless "
            "across batches, and keeps its engine loaded between them);\n"
            "  - set output_dtype to float16, which halves the figure above;\n"
            "  - lower the output resolution, or the frame count per run."
        ) from e


def iter_upscaled_frames(images, quality, scale, custom_width, custom_height,
                         align="even", dtype=torch.float32, keep_loaded=True):
    """Yield ``(out_width, out_height)``, then the upscaled frames one by one.

    A generator rather than a batch, so a caller that streams the result to an
    encoder never has to hold more than a single frame.
    """
    if images is None or images.shape[0] == 0:
        raise RuntimeError("No images to upscale")
    if images.shape[-1] == 4:
        images = images[..., :3]  # VSR takes RGB; an alpha channel is dropped
    elif images.shape[-1] != 3:
        raise RuntimeError(f"Expected an RGB IMAGE batch, got {images.shape[-1]} channels")

    frame_count, height, width, _ = images.shape
    out_width, out_height = resolve_output_size(width, height, quality, scale,
                                                custom_width, custom_height, align)

    device = model_management.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError("RTX Video Super Resolution needs a CUDA device, "
                           f"but ComfyUI is running on {device}.")
    device_index = device.index if device.index is not None else torch.cuda.current_device()

    effect = _SESSION.get(quality, out_width, out_height, device_index)
    stream_ptr = torch.cuda.current_stream(device).cuda_stream

    yield (out_width, out_height)

    pbar = ProgressBar(frame_count)
    try:
        for i in range(frame_count):
            model_management.throw_exception_if_processing_interrupted()
            frame = (images[i].to(device=device, dtype=torch.float32, non_blocking=True)
                     .clamp(0.0, 1.0).permute(2, 0, 1).contiguous())
            result = effect.run(frame, stream_ptr=stream_ptr)
            # the capsule points at SDK-owned memory, clone before the next run()
            upscaled = torch.from_dlpack(result.image).clone()
            # cast on the GPU so the copy over PCIe is already the smaller one
            yield upscaled.permute(1, 2, 0).clamp(0.0, 1.0).to(dtype).cpu()
            del frame, result, upscaled
            pbar.update(1)
    finally:
        if not keep_loaded:
            _SESSION.close()
        model_management.soft_empty_cache()


def upscale_images(images, quality, scale, custom_width, custom_height,
                   align="even", output_dtype="float32", keep_loaded=True):
    dtype = OUTPUT_DTYPES.get(output_dtype, torch.float32)
    frames = iter_upscaled_frames(images, quality, scale, custom_width, custom_height,
                                  align=align, dtype=dtype, keep_loaded=keep_loaded)
    out_width, out_height = next(frames)
    output = allocate_output(images.shape[0], out_height, out_width, dtype)
    for i, frame in enumerate(frames):
        output[i] = frame
    return output


class RTXVideoUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "quality": (get_quality_options(), {
                    "default": "HIGH",
                    "tooltip": "BICUBIC/LOW..ULTRA upscale, HIGHBITRATE_* skip artifact "
                               "removal for clean sources, DENOISE_*/DEBLUR_* enhance at "
                               "source resolution (scale is ignored for those).",
                }),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                                    "tooltip": "Used when custom_width and custom_height are 0."}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8,
                                         "tooltip": "0 = derive from scale. Set one of the two "
                                                    "to scale by aspect ratio."}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Passed straight through, untouched."}),
                "align": (list(ALIGN_OPTIONS), {
                    "default": "even",
                    "tooltip": "Round the output size. The SDK takes any size, but h264/h265 "
                               "need even dimensions and a latent round trip wants /8.",
                }),
                "output_dtype": (list(OUTPUT_DTYPES), {
                    "default": "float32",
                    "tooltip": "float16 halves the RAM the upscaled batch needs, at 11 bits "
                               "of mantissa instead of 24 -- invisible on 8-bit video.",
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the VSR engine in VRAM between runs. Turn off to free "
                               "it after each execution, at the cost of a reload next time.",
                }),
            },
        }

    CATEGORY = "Helpers 🧰"

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("IMAGE", "audio")

    FUNCTION = "upscale"

    DESCRIPTION = ("NVIDIA RTX Video Super Resolution applied to an IMAGE batch, in place "
                   "in the workflow. Audio is passed through unchanged.")

    def upscale(self, images, quality, scale, custom_width, custom_height,
                audio=None, align="even", output_dtype="float32", keep_model_loaded=True):
        result = upscale_images(images, quality, scale, custom_width, custom_height,
                                align=align, output_dtype=output_dtype,
                                keep_loaded=keep_model_loaded)
        return (result, audio)


class RTXVideoUpscaleToFile:
    """The same upscale, encoded straight to a file instead of an IMAGE batch.

    An IMAGE output has to exist as one contiguous tensor: 1000 frames of
    1280x704 is 10.5 GiB of RAM before anything downstream touches it. This
    variant hands each frame to the encoder and drops it, so the upscale itself
    costs one frame, and the file on disk is the result.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "quality": (get_quality_options(), {"default": "HIGH"}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                                    "tooltip": "Used when custom_width and custom_height are 0."}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0,
                                         "step": 0.01,
                                         "tooltip": "Feed this from the loader's video_info "
                                                    "to keep the original timing."}),
                "filename_prefix": ("STRING", {"default": "rtx/upscaled"}),
                "format": (list(VIDEO_FORMATS), {"default": "mp4"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1,
                                "tooltip": "h264 quality: 0 lossless, 18 visually lossless, "
                                           "23 default, 51 worst. Lower means bigger."}),
                "save_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: written to output/. Off: written to temp/, so it still "
                               "plays in the preview below but isn't kept.",
                }),
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embed the prompt and workflow in the file, the way ComfyUI's "
                               "own save nodes do. Reliable in mp4, best-effort in avi.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Muxed into the file. Leave unconnected for "
                                               "a video with no audio track."}),
                "align": (list(ALIGN_OPTIONS), {"default": "even"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "Helpers 🧰"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)

    FUNCTION = "upscale_to_file"
    OUTPUT_NODE = True

    DESCRIPTION = ("NVIDIA RTX Video Super Resolution encoded straight to a video file, "
                   "one frame at a time, so a long upscale never has to fit in RAM.")

    def upscale_to_file(self, images, quality, scale, custom_width, custom_height,
                        frame_rate, filename_prefix, format, crf, save_output,
                        save_metadata, audio=None, align="even", keep_model_loaded=True,
                        prompt=None, extra_pnginfo=None):
        # imported here so a missing PyAV only breaks this node, not the pack
        from .video_encode import CONTAINER_FORMATS, write_video

        frames = iter_upscaled_frames(images, quality, scale, custom_width, custom_height,
                                      align=align, keep_loaded=keep_model_loaded)
        out_width, out_height = next(frames)

        folder_type = "output" if save_output else "temp"
        base_dir = (folder_paths.get_output_directory() if save_output
                    else folder_paths.get_temp_directory())
        full_output_folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, base_dir, out_width, out_height)
        os.makedirs(full_output_folder, exist_ok=True)
        file = f"{name}_{counter:05}_.{CONTAINER_FORMATS[format]['extension']}"
        path = os.path.join(full_output_folder, file)

        metadata = {}
        if save_metadata:
            if prompt is not None:
                metadata["prompt"] = prompt
            for key, value in (extra_pnginfo or {}).items():
                metadata[key] = value

        write_video(path, frames, out_width, out_height, frame_rate, crf=crf,
                    container_format=format, audio=audio, metadata=metadata,
                    total_frames=images.shape[0])

        return {
            "ui": {"images": [{"filename": file, "subfolder": subfolder, "type": folder_type}],
                   "animated": (True,)},
            "result": (path,),
        }


class LegacyRTXVideoUpscale(RTXVideoUpscale):
    """The pre-CMDR_ node id, so workflows saved with it still resolve."""
    DEPRECATED = True


ARTIFACT_STRENGTHS = {
    "LOW": "DENOISE_LOW", "MED": "DENOISE_MEDIUM",
    "HIGH": "DENOISE_HIGH", "ULTRA": "DENOISE_ULTRA",
}
GREEN_SCREEN_MODES = {
    "QUALITY (chairs foreground)": 0,
    "PERFORMANCE (chairs foreground)": 1,
    "QUALITY (chairs background)": 2,
    "PERFORMANCE (chairs background)": 3,
}

# Separate from the existing upscaler: cleanup cannot change its cached settings.
_ARTIFACT_SESSION = _VSRSession()
_VFX_LOCK = threading.RLock()
_NATIVE_SESSIONS = {}


def get_artifact_strengths():
    try:
        available = {q.name for q in get_quality_enum()}
    except Exception:
        return list(ARTIFACT_STRENGTHS)
    return [s for s, q in ARTIFACT_STRENGTHS.items() if q in available]


def validate_vfx_images(images):
    if (images is None or images.ndim != 4 or any(d == 0 for d in images.shape)
            or images.shape[-1] not in (3, 4)):
        raise ValueError("Expected a non-empty IMAGE batch [N, H, W, 3 or 4]")
    return images[..., :3]


def vfx_device():
    device = model_management.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError(f"RTX VFX requires CUDA; ComfyUI is using {device}.")
    return torch.device("cuda", device.index if device.index is not None
                        else torch.cuda.current_device())


def validate_vfx_mask(mask, images):
    if mask is None or mask.ndim not in (2, 3):
        raise ValueError("Expected MASK [H, W] or [N, H, W]")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape[1:]) != tuple(images.shape[1:3]):
        raise ValueError("MASK dimensions must match IMAGE dimensions (no automatic resize)")
    if mask.shape[0] not in (1, images.shape[0]):
        raise ValueError("Provide one mask to broadcast, or exactly one mask per image")
    return mask


def process_vfx(images, kind, setting, output_dtype, keep_loaded, mask=None, temporal=True):
    images = validate_vfx_images(images)
    count, height, width, _ = images.shape
    dtype = OUTPUT_DTYPES[output_dtype]
    if kind == "GreenScreen" and (width < 512 or height < 288):
        raise ValueError("NVIDIA AI Green Screen requires at least 512x288; input is not resized")
    if kind == "BackgroundBlur":
        mask = validate_vfx_mask(mask, images)
    device = vfx_device()
    # Serialise mutable SDK engines across node instances and queued executions.
    with _VFX_LOCK, torch.cuda.device(device):
        if kind == "Artifact":
            session = _ARTIFACT_SESSION
        else:
            from .rtx_vfx_native import NativeSession
            session = _NATIVE_SESSIONS.setdefault(kind, NativeSession(kind))
        failed = True
        try:
            output = allocate_output(count, height, width, dtype)
            masks = (torch.empty((count, height, width), dtype=torch.float32)
                     if kind == "GreenScreen" else None)
            if kind == "Artifact":
                try:
                    effect = session.get(setting, width, height, device.index)
                except KeyError as exc:
                    raise RuntimeError(f"Installed nvvfx lacks {setting}; update nvidia-vfx "
                                       "to use same-resolution artifact reduction.") from exc
            else:
                effect = session.get(device, height, width, setting)
                effect.bind_stream()
                effect.reset()
            pbar = ProgressBar(count)
            for i in range(count):
                model_management.throw_exception_if_processing_interrupted()
                frame = images[i].to(device=device, dtype=torch.float32).clamp(0, 1)
                if kind == "Artifact":
                    source = frame.permute(2, 0, 1).contiguous()
                    result = effect.run(source, stream_ptr=torch.cuda.current_stream(device).cuda_stream)
                    processed = torch.from_dlpack(result.image).clone().permute(1, 2, 0)
                elif kind == "GreenScreen":
                    if not temporal and i:
                        effect.reset()
                    alpha = effect.run(frame)
                    if tuple(alpha.shape) != (height, width):
                        raise RuntimeError("NVIDIA VFX returned a mask with unexpected dimensions")
                    masks[i].copy_(alpha.cpu())
                    processed = frame * alpha.unsqueeze(-1)
                else:
                    alpha = mask[0 if mask.shape[0] == 1 else i].to(device=device, dtype=torch.float32)
                    processed = effect.run(frame, alpha)
                if tuple(processed.shape) != (height, width, 3):
                    raise RuntimeError("NVIDIA VFX changed image dimensions; refusing resized output")
                output[i].copy_(processed.clamp(0, 1).to(dtype).cpu())
                pbar.update(1)
            failed = False
            return (output, masks) if masks is not None else output
        finally:
            try:
                if failed or not keep_loaded:
                    session.close()
            finally:
                model_management.soft_empty_cache()


def vfx_common_inputs():
    return {
        "output_dtype": (list(OUTPUT_DTYPES), {"default": "float16"}),
        "keep_loaded": ("BOOLEAN", {"default": True}),
    }


class RTXArtifactReduction:
    @classmethod
    def INPUT_TYPES(cls):
        strengths = get_artifact_strengths() or list(ARTIFACT_STRENGTHS)
        return {"required": {
            "images": ("IMAGE",),
            "strength": (strengths, {"default": "LOW" if "LOW" in strengths else strengths[0]}),
            **vfx_common_inputs(),
        }, "optional": {"audio": ("AUDIO",)}}

    CATEGORY = "Helpers 🧰"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "reduce"
    DESCRIPTION = ("Same-resolution NVIDIA DENOISE artifact cleanup. Never scales or aligns "
                   "dimensions. Audio passes through unchanged.")

    def reduce(self, images, strength="LOW", output_dtype="float16", keep_loaded=True, audio=None):
        result = process_vfx(images, "Artifact", ARTIFACT_STRENGTHS[strength], output_dtype, keep_loaded)
        return result, audio


class RTXAIGreenScreen:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "mode": (list(GREEN_SCREEN_MODES), {"default": "QUALITY (chairs foreground)"}),
            "temporal": ("BOOLEAN", {"default": True, "tooltip":
                "On for ordered video frames; off for independent images. State resets each execution."}),
            **vfx_common_inputs(),
        }}

    CATEGORY = "Helpers 🧰"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("foreground", "mask")
    FUNCTION = "segment"
    DESCRIPTION = ("NVIDIA human segmentation. Foreground is RGB over black; the soft MASK "
                   "is white for foreground, black for background. Minimum input: 512x288.")

    def segment(self, images, mode="QUALITY (chairs foreground)", temporal=True,
                output_dtype="float16", keep_loaded=True):
        return process_vfx(images, "GreenScreen", GREEN_SCREEN_MODES[mode],
                           output_dtype, keep_loaded, temporal=temporal)


class RTXBackgroundBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "mask": ("MASK", {"tooltip": "White protects foreground; black blurs background."}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            **vfx_common_inputs(),
        }}

    CATEGORY = "Helpers 🧰"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "blur"
    DESCRIPTION = "NVIDIA background blur. Connect the original IMAGE and foreground MASK."

    def blur(self, images, mask, strength=0.5, output_dtype="float16", keep_loaded=True):
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Blur strength must be between 0 and 1")
        return (process_vfx(images, "BackgroundBlur", strength, output_dtype, keep_loaded, mask=mask),)


NODE_CLASS_MAPPINGS = {
    "CMDR_RTXArtifactReduction": RTXArtifactReduction,
    "CMDR_RTXAIGreenScreen": RTXAIGreenScreen,
    "CMDR_RTXBackgroundBlur": RTXBackgroundBlur,
    "CMDR_RTXVideoUpscale": RTXVideoUpscale,
    "CMDR_RTXVideoUpscaleToFile": RTXVideoUpscaleToFile,
    "Helpers_RTXVideoUpscale": LegacyRTXVideoUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CMDR_RTXArtifactReduction": "RTX Artifact Reduction",
    "CMDR_RTXAIGreenScreen": "RTX AI Green Screen",
    "CMDR_RTXBackgroundBlur": "RTX Background Blur",
    "CMDR_RTXVideoUpscale": "RTX Video Upscale (IMAGE) 🧰",
    "CMDR_RTXVideoUpscaleToFile": "RTX Video Upscale (to file) 🧰",
    "Helpers_RTXVideoUpscale": "RTX Video Upscale (IMAGE) 🧰 (old id)",
}
