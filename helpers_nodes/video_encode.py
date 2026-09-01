"""Streaming video writer built on PyAV.

PyAV already ships with ComfyUI (core's Save Video nodes use it), so there is no
ffmpeg binary to find on PATH and no subprocess pipe to babysit.

The point of this module is that it consumes an *iterator* of frames: nothing
here ever holds more than one frame, which is what lets a node encode a long
upscale without materialising the result as one big IMAGE tensor.
"""
import json
from fractions import Fraction

import av
import av.audio.fifo
import numpy as np
import torch

# Both containers get h264: it keeps `crf` meaningful across formats, and avi
# with h264 plays anywhere the last two decades of software is installed.
CONTAINER_FORMATS = {
    "mp4": {"codec": "libx264", "audio_codec": "aac", "extension": "mp4"},
    "avi": {"codec": "libx264", "audio_codec": "mp3", "extension": "avi"},
}


def frame_to_uint8(frame):
    """float [0,1] HWC tensor/array -> contiguous uint8 RGB, as PyAV wants it."""
    if isinstance(frame, torch.Tensor):
        frame = frame[..., :3].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    else:
        frame = np.clip(np.asarray(frame)[..., :3], 0.0, 1.0)
        frame = (frame * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(frame)


def add_audio(container, audio, audio_codec):
    """Add an audio stream, or return None when there is nothing to add."""
    if audio is None:
        return None, None
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 44100))
    if waveform is None or waveform.numel() == 0:
        return None, None

    samples = waveform[0] if waveform.dim() == 3 else waveform  # (channels, n)
    samples = samples.to(torch.float32).clamp(-1.0, 1.0).cpu().numpy()
    channels = samples.shape[0]
    layout = "mono" if channels == 1 else ("stereo" if channels == 2 else f"{channels}c")

    stream = container.add_stream(audio_codec, rate=sample_rate)
    stream.layout = layout

    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples), format="fltp",
                                       layout=layout)
    frame.sample_rate = sample_rate
    return stream, frame


def encode_audio(container, stream, frame):
    """Feed the encoder through a FIFO: it wants its own fixed frame size."""
    fifo = av.audio.fifo.AudioFifo()
    fifo.write(frame)
    frame_size = stream.frame_size or 1024
    while True:
        chunk = fifo.read(frame_size)
        if chunk is None:
            break
        for packet in stream.encode(chunk):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)


def write_video(path, frames, width, height, fps, crf=19, container_format="mp4",
                audio=None, metadata=None, total_frames=None, on_frame=None):
    """Encode ``frames`` (an iterable of float [0,1] HWC frames) to ``path``.

    ``metadata`` values are json-dumped, matching what ComfyUI's own save nodes
    embed, so the workflow can be recovered from the file.
    """
    spec = CONTAINER_FORMATS[container_format]

    # mp4's muxer drops tags it doesn't recognise unless told otherwise
    options = {"movflags": "use_metadata_tags"} if container_format == "mp4" else {}
    container = av.open(path, mode="w", options=options)
    try:
        for key, value in (metadata or {}).items():
            if value is not None:
                container.metadata[key] = json.dumps(value)

        stream = container.add_stream(spec["codec"], rate=Fraction(round(fps * 1000), 1000))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(int(crf))}

        audio_stream, audio_frame = add_audio(container, audio, spec["audio_codec"])

        written = 0
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame_to_uint8(frame), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
            written += 1
            if on_frame is not None:
                on_frame(written, total_frames)
        for packet in stream.encode(None):
            container.mux(packet)

        if audio_stream is not None:
            encode_audio(container, audio_stream, audio_frame)
    finally:
        container.close()
    return written
