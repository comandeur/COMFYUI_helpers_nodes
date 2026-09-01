"""NVIDIA RTX Video Super Resolution, straight on IMAGE tensors.

ComfyUI-RTX-Video-Suite wraps the same SDK but works on files: it reads a video
off disk and writes an upscaled one back. That means leaving the workflow,
re-encoding, and a round trip through ffmpeg every time.

This node stays inside the graph -- IMAGE in, IMAGE out, AUDIO carried through
untouched -- so it can sit between any two video nodes. The frames are handed to
the SDK on the GPU (DLPack, no host copy in between) one at a time, so VRAM cost
is one input frame plus one output frame regardless of batch length.

Requires the NVIDIA Video Effects (VFX) SDK, i.e. an importable ``nvvfx``, and an
RTX GPU. Without it the node still registers and only fails when executed.
"""
import torch

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


def upscale_images(images, quality, scale, custom_width, custom_height,
                   align="even", keep_loaded=True):
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

    output = torch.empty((frame_count, out_height, out_width, 3), dtype=torch.float32)
    pbar = ProgressBar(frame_count)
    try:
        for i in range(frame_count):
            model_management.throw_exception_if_processing_interrupted()
            frame = (images[i].to(device=device, dtype=torch.float32, non_blocking=True)
                     .clamp(0.0, 1.0).permute(2, 0, 1).contiguous())
            result = effect.run(frame, stream_ptr=stream_ptr)
            # the capsule points at SDK-owned memory, clone before the next run()
            upscaled = torch.from_dlpack(result.image).clone()
            output[i] = upscaled.permute(1, 2, 0).clamp(0.0, 1.0).cpu()
            del frame, result, upscaled
            pbar.update(1)
    finally:
        if not keep_loaded:
            _SESSION.close()
        model_management.soft_empty_cache()

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
                audio=None, align="even", keep_model_loaded=True):
        result = upscale_images(images, quality, scale, custom_width, custom_height,
                                align=align, keep_loaded=keep_model_loaded)
        return (result, audio)


NODE_CLASS_MAPPINGS = {
    "Helpers_RTXVideoUpscale": RTXVideoUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Helpers_RTXVideoUpscale": "RTX Video Upscale (IMAGE) 🧰",
}
