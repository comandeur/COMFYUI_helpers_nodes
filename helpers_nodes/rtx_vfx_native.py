"""Small lazy ctypes bridge to the native NVIDIA VFX SDK.

Green Screen and BackgroundBlur are not exposed by nvidia-vfx's public Python
API. ABI and selectors follow NVIDIA's nvCVImage.h / nvVideoEffects.h.
Buffers belong to torch, never to the SDK; all calls use torch's CUDA stream.
"""
import ctypes as ct
import os
from pathlib import Path

import torch


class NvCVImage(ct.Structure):
    _fields_ = [
        ("width", ct.c_uint), ("height", ct.c_uint), ("pitch", ct.c_int),
        ("pixelFormat", ct.c_int), ("componentType", ct.c_int),
        ("pixelBytes", ct.c_ubyte), ("componentBytes", ct.c_ubyte),
        ("numComponents", ct.c_ubyte), ("planar", ct.c_ubyte),
        ("gpuMem", ct.c_ubyte), ("colorspace", ct.c_ubyte),
        ("reserved", ct.c_ubyte * 2), ("pixels", ct.c_void_p),
        ("deletePtr", ct.c_void_p), ("deleteProc", ct.c_void_p),
        ("bufferBytes", ct.c_ulonglong),
    ]


def image_descriptor(tensor):
    if tensor.dtype != torch.uint8 or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("Native VFX requires contiguous CUDA uint8 buffers")
    channels = 1 if tensor.ndim == 2 else tensor.shape[2]
    if tensor.ndim not in (2, 3) or channels not in (1, 3):
        raise ValueError("Expected HxW alpha or HxWx3 BGR buffer")
    desc = NvCVImage()
    desc.height, desc.width = tensor.shape[:2]
    desc.pitch = tensor.stride(0)
    desc.pixelFormat = 2 if channels == 1 else 5  # NVCV_A / NVCV_BGR
    desc.componentType = 1  # NVCV_U8
    desc.pixelBytes = desc.numComponents = channels
    desc.componentBytes = desc.gpuMem = 1
    desc.pixels = tensor.data_ptr()
    desc.bufferBytes = tensor.numel()
    return desc


class NativeAPI:
    def __init__(self):
        self.dll_dirs = []
        root = os.environ.get("NV_VIDEO_EFFECTS_PATH")
        if not root and os.name == "nt":
            root = str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) /
                       "NVIDIA Corporation/NVIDIA Video Effects")
        name = "NVVideoEffects.dll" if os.name == "nt" else "libNVVideoEffects.so"
        candidates = [Path(root) / name, Path(root) / "bin" / name,
                      Path(root) / "lib" / name] if root else []
        library = next((p for p in candidates if p.is_file()), None)
        if library and os.name == "nt":
            self.dll_dirs.append(os.add_dll_directory(str(library.parent.resolve())))
        try:
            self.lib = ct.CDLL(str(library) if library else name)
        except OSError as exc:
            raise RuntimeError(
                "NVIDIA native VFX SDK unavailable. Install the Green Screen and "
                "Background Blur features and their dependencies, set NV_VIDEO_EFFECTS_PATH "
                "to the SDK library directory and NVVFX_MODEL_DIR to its models directory, "
                "then restart ComfyUI. Installing nvidia-vfx alone is not sufficient."
            ) from exc
        self.model_dir = os.environ.get("NVVFX_MODEL_DIR")
        if not self.model_dir and root and (Path(root) / "models").is_dir():
            self.model_dir = str(Path(root) / "models")
        ptr, string = ct.c_void_p, ct.c_char_p
        signatures = {
            "CreateEffect": [string, ct.POINTER(ptr)], "DestroyEffect": [ptr],
            "SetU32": [ptr, string, ct.c_uint], "SetF32": [ptr, string, ct.c_float],
            "SetString": [ptr, string, string], "SetCudaStream": [ptr, string, ptr],
            "SetImage": [ptr, string, ct.POINTER(NvCVImage)],
            "Load": [ptr], "Run": [ptr, ct.c_int],
            "AllocateState": [ptr, ct.POINTER(ptr)], "DeallocateState": [ptr, ptr],
            "ResetState": [ptr, ptr],
            "SetStateObjectHandleArray": [ptr, string, ct.POINTER(ptr)],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, "NvVFX_" + name)
            fn.argtypes = args
            fn.restype = None if name == "DestroyEffect" else ct.c_int

    def call(self, name, *args):
        status = getattr(self.lib, "NvVFX_" + name)(*args)
        if status:
            raise RuntimeError(f"NvVFX_{name} failed (status {status}). Check the installed "
                               "VFX features, models, GPU driver and input dimensions.")


class NativeSession:
    """One effect plus fixed-size frame buffers, keyed by GPU, size and settings."""
    def __init__(self, kind):
        self.kind = kind
        self.api = None
        self.handle = ct.c_void_p()
        self.state = ct.c_void_p()
        self.key = None
        self.src = self.dst = self.mask = None
        self.device = None

    def get(self, device, height, width, setting):
        key = (str(device), height, width, setting)
        if self.key == key:
            return self
        self.close()
        self.device = device
        try:
            self.api = self.api or NativeAPI()
            self.api.call("CreateEffect", self.kind.encode(), ct.byref(self.handle))
            if self.api.model_dir:
                self.api.call("SetString", self.handle, b"ModelDir",
                              os.fsencode(self.api.model_dir))
            self.bind_stream()
            self.src = torch.empty((height, width, 3), device=device, dtype=torch.uint8)
            self.mask = torch.empty((height, width), device=device, dtype=torch.uint8)
            self.dst = (self.mask if self.kind == "GreenScreen" else torch.empty_like(self.src))
            for selector, tensor in ((b"SrcImage0", self.src), (b"DstImage0", self.dst)):
                self.api.call("SetImage", self.handle, selector, ct.byref(image_descriptor(tensor)))
            if self.kind == "GreenScreen":
                self.api.call("SetU32", self.handle, b"Mode", setting)
                self.api.call("SetU32", self.handle, b"BatchSize", 1)
                self.api.call("SetU32", self.handle, b"ModelBatch", 1)
                self.api.call("SetU32", self.handle, b"MaxInputWidth", width)
                self.api.call("SetU32", self.handle, b"MaxInputHeight", height)
                self.api.call("SetU32", self.handle, b"MaxNumberStreams", 1)
            else:
                self.api.call("SetImage", self.handle, b"SrcImage1", ct.byref(image_descriptor(self.mask)))
                self.api.call("SetF32", self.handle, b"Strength", setting)
            self.api.call("Load", self.handle)
            if self.kind == "GreenScreen":
                self.api.call("AllocateState", self.handle, ct.byref(self.state))
                self.api.call("SetStateObjectHandleArray", self.handle, b"State", ct.byref(self.state))
            self.key = key
            return self
        except BaseException:
            self.close()
            raise

    def bind_stream(self):
        stream = torch.cuda.current_stream(self.device)
        self.api.call("SetCudaStream", self.handle, b"CudaStream", stream.cuda_stream)

    def reset(self):
        if self.state.value:
            self.api.call("ResetState", self.handle, self.state)

    def run(self, frame, mask=None):
        self.src.copy_(frame.flip(-1).clamp(0, 1).mul(255).round().to(torch.uint8))
        if mask is not None:
            self.mask.copy_(mask.clamp(0, 1).mul(255).round().to(torch.uint8))
        self.api.call("Run", self.handle, 0)
        # Synchronize before buffers can be reused or released, including on errors.
        torch.cuda.current_stream(self.device).synchronize()
        if self.kind == "GreenScreen":
            return self.dst.float().div_(255)
        return self.dst.flip(-1).float().div_(255)

    def close(self):
        try:
            if self.handle.value:
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize(self.device)
                    try:
                        if self.state.value:
                            self.api.call("DeallocateState", self.handle, self.state)
                    finally:
                        self.api.call("DestroyEffect", self.handle)
        finally:
            self.handle = ct.c_void_p()
            self.state = ct.c_void_p()
            self.key = None
            self.src = self.dst = self.mask = None
            self.device = None
