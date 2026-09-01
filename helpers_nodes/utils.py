"""Shared helpers for the COMFYUI_helpers_nodes pack."""
import hashlib
import os

import numpy as np
import torch

BIGMAX = (2 ** 53 - 1)
DIMMAX = 8192


class MultiInput(str):
    """A type string that also accepts a list of alternative types.

    Same trick VideoHelperSuite uses, so our sockets stay link-compatible with
    theirs (an ``imageOrLatent`` output can feed an ``IMAGE`` or ``LATENT`` input).
    """

    def __new__(cls, string, allowed_types="*"):
        res = super().__new__(cls, string)
        res.allowed_types = allowed_types
        return res

    def __ne__(self, other):
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types


imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])


def calculate_file_hash(filename: str) -> str:
    """Cheap IS_CHANGED fingerprint: path + mtime rather than file contents."""
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()


def strip_path(path: str) -> str:
    path = path.strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path


def target_size(width, height, custom_width, custom_height, downscale_ratio=8):
    """Resolve the output size, rounded to a multiple of ``downscale_ratio``."""
    if downscale_ratio is None:
        downscale_ratio = 8
    if custom_width == 0 and custom_height == 0:
        pass
    elif custom_height == 0:
        height *= custom_width / width
        width = custom_width
    elif custom_width == 0:
        width *= custom_height / height
        height = custom_height
    else:
        width = custom_width
        height = custom_height
    width = int(width / downscale_ratio + 0.5) * downscale_ratio
    height = int(height / downscale_ratio + 0.5) * downscale_ratio
    return (max(downscale_ratio, width), max(downscale_ratio, height))


SILENT_AUDIO_SAMPLE_RATE = 44100
SILENT_AUDIO_CHANNELS = 2


def make_null_audio(mode: str, duration: float = 0.0,
                    sample_rate: int = SILENT_AUDIO_SAMPLE_RATE):
    """GIF/WebP carry no audio, so build whatever stand-in the user asked for.

    ``none``   -> a real ``None``. Nothing is encoded downstream, but a node that
                  *requires* an AUDIO input will complain.
    ``silent`` -> a valid, silent AUDIO dict as long as the loaded animation, so
                  it can be muxed / previewed / saved like any other audio.
    """
    if mode == "silent":
        samples = max(1, int(round(duration * sample_rate)))
        return {
            "waveform": torch.zeros((1, SILENT_AUDIO_CHANNELS, samples), dtype=torch.float32),
            "sample_rate": sample_rate,
        }
    return None


def batched(iterable, n):
    """itertools.batched, available on every python ComfyUI supports."""
    import itertools
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def batched_vae_encode(images, vae, frames_per_batch):
    for batch in batched(images, frames_per_batch):
        image_batch = torch.from_numpy(np.array(batch))
        yield from vae.encode(image_batch).numpy()
