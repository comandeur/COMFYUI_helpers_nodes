"""Load GIF / animated WebP (and APNG) as an image batch.

Deliberately shaped like VideoHelperSuite's "Load Video (Upload)": same widgets,
same four outputs, same socket types, so it drops into existing video workflows.
The differences are all consequences of the source format:

  * frames are decoded with Pillow, which understands GIF disposal methods and
    WebP alpha (OpenCV/ffmpeg either mangle or drop those),
  * per-frame delays are honoured -- GIF/WebP timing is variable, not a constant
    fps -- so ``force_rate`` resamples against the real timeline,
  * there is no audio stream, so the ``audio`` output is a null/silent stand-in.
"""
import itertools
import os

import numpy as np
import psutil
import torch
from PIL import Image, ImageSequence

import folder_paths
from comfy.utils import ProgressBar, common_upscale

from .formats import get_format, get_load_formats
from .utils import (BIGMAX, DIMMAX, batched, batched_vae_encode,
                    calculate_file_hash, floatOrInt, imageOrLatent,
                    make_null_audio, strip_path, target_size)

ANIMATION_EXTENSIONS = ['gif', 'webp', 'apng']

# Browsers (and every GIF written for one) treat a delay below 2/100s as
# "unspecified" and render it at 10fps. Match that instead of reporting a
# nonsensical 1000fps source.
MIN_FRAME_DELAY = 0.02
DEFAULT_FRAME_DELAY = 0.1

ALPHA_MODES = ["composite_black", "composite_white", "keep_alpha"]
AUDIO_MODES = ["none", "silent"]


def webp_frame_durations(path):
    """Read the per-frame delays (in milliseconds) out of the WebP container.

    Pillow only fills ``info['duration']`` for a WebP frame once that frame has
    been *decoded*, so probing through Pillow would decode the whole animation
    twice. The ANMF chunk headers carry the delay, so read those instead.
    Returns None if the file isn't a RIFF/WEBP we can walk.
    """
    durations = []
    try:
        with open(path, 'rb') as f:
            header = f.read(12)
            if len(header) < 12 or header[:4] != b'RIFF' or header[8:12] != b'WEBP':
                return None
            while True:
                chunk = f.read(8)
                if len(chunk) < 8:
                    break
                fourcc, size = chunk[:4], int.from_bytes(chunk[4:8], 'little')
                padded = size + (size & 1)
                if fourcc == b'ANMF':
                    payload = f.read(min(16, size))
                    if len(payload) < 16:
                        return None
                    durations.append(int.from_bytes(payload[12:15], 'little'))
                    f.seek(padded - len(payload), 1)
                else:
                    f.seek(padded, 1)
    except OSError:
        return None
    return durations or None


def probe_animation(path):
    """Return (width, height, [frame durations in seconds]).

    Walks the frames once so the node knows the real length and timeline before
    it starts loading pixels.
    """
    with Image.open(path) as img:
        width, height = img.size
        raw = None
        if (img.format or '').upper() == 'WEBP':
            raw = webp_frame_durations(path)
            if raw is not None and len(raw) != getattr(img, 'n_frames', 1):
                raw = None  # container walk disagrees with Pillow, don't trust it
        if raw is None:
            raw = []
            for frame in ImageSequence.Iterator(img):
                delay = frame.info.get('duration') or 0
                if not delay:
                    # some plugins only publish the delay once the frame is decoded
                    frame.load()
                    delay = frame.info.get('duration') or 0
                raw.append(delay)

    durations = []
    for delay in raw:
        try:
            delay = float(delay) / 1000.0
        except (TypeError, ValueError):
            delay = 0.0
        if delay < MIN_FRAME_DELAY:
            # 0 (and the 1/100s most encoders write) means "as fast as possible";
            # browsers render those at 10fps, so match that instead of reporting
            # a nonsensical source rate.
            delay = DEFAULT_FRAME_DELAY
        durations.append(delay)
    if not durations:
        raise ValueError(f"{path} contains no frames")
    return width, height, durations


def select_frame_indices(durations, force_rate, skip_first_frames,
                         select_every_nth, frame_load_cap):
    """Resolve which source frames end up in the batch, in load order.

    Order matches VideoHelperSuite: rate conversion, then skip, then every-nth,
    then the cap.
    """
    frame_count = len(durations)
    if force_rate == 0:
        indices = list(range(frame_count))
        base_frame_time = sum(durations) / frame_count
    else:
        base_frame_time = 1.0 / force_rate
        starts = list(itertools.accumulate([0.0] + durations[:-1]))
        total = sum(durations)
        indices = []
        cursor = 0
        sample_time = 0.0
        while sample_time < total - 1e-9:
            while cursor + 1 < frame_count and starts[cursor + 1] <= sample_time + 1e-9:
                cursor += 1
            indices.append(cursor)
            sample_time += base_frame_time

    indices = indices[skip_first_frames:]
    indices = indices[::select_every_nth]
    if frame_load_cap > 0:
        indices = indices[:frame_load_cap]
    return indices, base_frame_time


def pil_frame_generator(image, force_rate, frame_load_cap, skip_first_frames,
                        select_every_nth, alpha="composite_black",
                        meta_batch=None, unique_id=None):
    width, height, durations = probe_animation(image)
    source_frames = len(durations)
    duration = sum(durations)
    fps = source_frames / duration if duration else 0.0

    indices, base_frame_time = select_frame_indices(
        durations, force_rate, skip_first_frames, select_every_nth, frame_load_cap)
    keep_alpha = alpha == "keep_alpha"

    yield (width, height, fps, duration, source_frames, base_frame_time,
           len(indices), keep_alpha)

    if not indices:
        return

    pbar = ProgressBar(len(indices))
    background = 1.0 if alpha == "composite_white" else 0.0
    img = Image.open(image)
    try:
        loaded_index = None
        frame = None
        for emitted, index in enumerate(indices):
            if index != loaded_index:
                img.seek(index)
                # Pillow applies GIF disposal / WebP blending while seeking
                # forward, so this RGBA canvas is already composited.
                rgba = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
                if keep_alpha:
                    frame = np.ascontiguousarray(rgba)
                else:
                    a = rgba[:, :, 3:4]
                    frame = np.ascontiguousarray(rgba[:, :, :3] * a + background * (1.0 - a))
                loaded_index = index
            pbar.update_absolute(emitted + 1, len(indices))
            yield frame
    finally:
        img.close()
    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id, None)
        meta_batch.has_closed_inputs = True


def resized_frame_generator(custom_width, custom_height, downscale_ratio, **kwargs):
    gen = pil_frame_generator(**kwargs)
    (width, height, fps, duration, source_frames, base_frame_time,
     yieldable_frames, keep_alpha) = next(gen)
    channels = 4 if keep_alpha else 3

    if custom_width != 0 or custom_height != 0 or downscale_ratio is not None:
        new_width, new_height = target_size(width, height, custom_width,
                                            custom_height, downscale_ratio)
    else:
        new_width, new_height = width, height

    yield (width, height, fps, duration, source_frames, base_frame_time,
           yieldable_frames, new_width, new_height, keep_alpha)

    if new_width == width and new_height == height:
        yield from gen
        return

    frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
    if kwargs.get('meta_batch') is not None:
        frames_per_batch = min(frames_per_batch, kwargs['meta_batch'].frames_per_batch)

    def rescale(frame_batch):
        s = torch.from_numpy(np.fromiter(
            frame_batch, np.dtype((np.float32, (height, width, channels)))))
        s = s.movedim(-1, 1)
        s = common_upscale(s, new_width, new_height, "lanczos", "center")
        return s.movedim(1, -1).numpy()

    yield from itertools.chain.from_iterable(map(rescale, batched(gen, frames_per_batch)))


def load_animation(meta_batch=None, unique_id=None, memory_limit_mb=None, vae=None,
                   format='None', alpha="composite_black", audio_output="none",
                   **kwargs):
    format = get_format(format)
    kwargs['image'] = strip_path(kwargs['image'])
    if vae is not None:
        downscale_ratio = getattr(vae, "downscale_ratio", 8)
    else:
        downscale_ratio = format.get('dim', (1,))[0]

    if meta_batch is None or unique_id not in meta_batch.inputs:
        gen = resized_frame_generator(meta_batch=meta_batch, unique_id=unique_id,
                                      downscale_ratio=downscale_ratio, alpha=alpha,
                                      **kwargs)
        info = next(gen)
        if meta_batch is not None:
            meta_batch.inputs[unique_id] = (gen, *info)
            if info[6]:
                meta_batch.total_frames = min(meta_batch.total_frames, info[6])
    else:
        gen, *info = meta_batch.inputs[unique_id]

    (width, height, fps, duration, source_frames, base_frame_time,
     yieldable_frames, new_width, new_height, keep_alpha) = info
    channels = 4 if keep_alpha else 3

    if memory_limit_mb is not None:
        memory_limit = int(memory_limit_mb) * 2 ** 20
    else:
        try:
            # leaves ~128MB unreserved, the same margin VideoHelperSuite uses
            memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2 ** 27
        except Exception:
            memory_limit = BIGMAX
    if vae is not None:
        # room to hold f32 frames, the latents, and decode wiggle room
        max_loadable_frames = int(memory_limit // (width * height * 3 * (4 + 4 + 1 / 10)))
    else:
        max_loadable_frames = int(memory_limit // (width * height * 3 * .1))
    max_loadable_frames = max(1, max_loadable_frames)

    original_gen = None
    if meta_batch is not None:
        if 'frames' in format:
            if meta_batch.frames_per_batch % format['frames'][0] != format['frames'][1]:
                error = (meta_batch.frames_per_batch - format['frames'][1]) % format['frames'][0]
                suggested = meta_batch.frames_per_batch - error
                if error > format['frames'][0] / 2:
                    suggested += format['frames'][0]
                raise RuntimeError("The chosen frames per batch is incompatible with "
                                   f"the selected format. Try {suggested}")
        if meta_batch.frames_per_batch > max_loadable_frames:
            raise RuntimeError(f"Meta Batch set to {meta_batch.frames_per_batch} frames "
                               f"but only {max_loadable_frames} can fit in memory")
        gen = itertools.islice(gen, meta_batch.frames_per_batch)
    else:
        original_gen = gen
        gen = itertools.islice(gen, max_loadable_frames)

    if vae is not None:
        if keep_alpha:
            raise RuntimeError("keep_alpha cannot be encoded by a VAE. "
                               "Use one of the composite alpha modes.")
        frames_per_batch = (1920 * 1080 * 16) // (new_width * new_height) or 1
        latent_channels = getattr(vae, 'latent_channels', 4)
        lw, lh = new_width // downscale_ratio, new_height // downscale_ratio
        images = torch.from_numpy(np.fromiter(
            batched_vae_encode(gen, vae, frames_per_batch),
            np.dtype((np.float32, (latent_channels, lh, lw)))))
    else:
        images = torch.from_numpy(np.fromiter(
            gen, np.dtype((np.float32, (new_height, new_width, channels)))))

    if original_gen is not None:
        try:
            next(original_gen)
            raise RuntimeError(f"Memory limit hit after loading {len(images)} frames. "
                               "Stopping execution.")
        except StopIteration:
            pass
    if len(images) == 0:
        raise RuntimeError("No frames generated")

    if 'frames' in format and len(images) % format['frames'][0] != format['frames'][1]:
        err_msg = (f"The number of frames loaded {len(images)}, does not match the "
                   "requirements of the currently selected format.")
        if len(format['frames']) > 2 and format['frames'][2]:
            raise RuntimeError(err_msg)
        div, mod = format['frames'][:2]
        images = images[:(len(images) - mod) // div * div + mod]

    target_frame_time = base_frame_time * kwargs.get('select_every_nth', 1)
    loaded_duration = len(images) * target_frame_time
    audio = make_null_audio(audio_output, loaded_duration)

    video_info = {
        "source_fps": fps,
        "source_frame_count": source_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1 / target_frame_time if target_frame_time else 0.0,
        "loaded_frame_count": len(images),
        "loaded_duration": loaded_duration,
        "loaded_width": new_width,
        "loaded_height": new_height,
    }
    if vae is None:
        return (images, len(images), audio, video_info)
    return ({"samples": images}, len(images), audio, video_info)


class LoadAnimationUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and file_parts[-1].lower() in ANIMATION_EXTENSIONS:
                    files.append(f)
        return {
            "required": {
                # The upload button, drag/drop target and animated preview are
                # added by web/js/helpers_upload.js: the core image_upload widget
                # can't offer .gif in the file dialog.
                "image": (sorted(files), {"tooltip": "GIF, animated WebP or APNG in ComfyUI/input"}),
                "force_rate": (floatOrInt, {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, "disable": 0}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, "disable": 0}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1, "disable": 0}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": get_load_formats(),
                "alpha": (ALPHA_MODES, {"default": "composite_black"}),
                "audio_output": (AUDIO_MODES, {
                    "default": "none",
                    "tooltip": "GIF/WebP carry no audio. 'none' outputs a null audio, "
                               "'silent' outputs a silent track as long as the animation.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    CATEGORY = "Helpers 🧰"

    RETURN_TYPES = (imageOrLatent, "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")

    FUNCTION = "load_animation"

    def load_animation(self, **kwargs):
        kwargs['image'] = folder_paths.get_annotated_filepath(strip_path(kwargs['image']))
        return load_animation(**kwargs)

    @classmethod
    def IS_CHANGED(s, image, **kwargs):
        return calculate_file_hash(folder_paths.get_annotated_filepath(image))

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid animation file: {}".format(image)
        return True


class LegacyLoadAnimationUpload(LoadAnimationUpload):
    """The pre-CMDR_ node id, so workflows saved with it still resolve.

    DEPRECATED keeps it out of the node search without breaking anything.
    """
    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "CMDR_LoadAnimationUpload": LoadAnimationUpload,
    "Helpers_LoadAnimationUpload": LegacyLoadAnimationUpload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CMDR_LoadAnimationUpload": "Load GIF/WebP (Upload) 🧰",
    "Helpers_LoadAnimationUpload": "Load GIF/WebP (Upload) 🧰 (old id)",
}
