"""CPU contract tests with fake SDK engines; run directly with Python + torch.

These exercise the real frame loop but do not validate NVIDIA GPU inference.
"""
import contextlib
import ctypes
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
# Load only this module, without the rest of ComfyUI or the node pack.
for name in ("folder_paths", "comfy", "comfy.model_management", "comfy.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
mm = sys.modules["comfy.model_management"]
mm.get_torch_device = lambda: torch.device("cpu")
mm.throw_exception_if_processing_interrupted = lambda: None
mm.soft_empty_cache = lambda: None
sys.modules["comfy.utils"].ProgressBar = lambda count: types.SimpleNamespace(update=lambda n: None)
package = types.ModuleType("rtx_test_helpers")
package.__path__ = [str(ROOT / "helpers_nodes")]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location("rtx_test_helpers.rtx_vsr_nodes", ROOT / "helpers_nodes/rtx_vsr_nodes.py")
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)
from rtx_test_helpers import rtx_vfx_native as native


class FakeVSR:
    def __init__(self):
        self.calls = []
        self.closed = 0
        self.bad_size = False

    def get(self, *args):
        self.calls.append(args)
        return self

    def run(self, frame, **kwargs):
        result = frame[:, :-1] if self.bad_size else frame.clone()
        return types.SimpleNamespace(image=result)

    def close(self):
        self.closed += 1


class FakeNative(FakeVSR):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.resets = 0
        self.seen_masks = []

    def bind_stream(self):
        pass

    def reset(self):
        self.resets += 1

    def run(self, frame, mask=None):
        if self.kind == "GreenScreen":
            alpha = torch.zeros(frame.shape[:2])
            alpha[:, :frame.shape[1] // 2] = 1
            return alpha
        self.seen_masks.append(mask.clone())
        return frame * mask.unsqueeze(-1)


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(nodes, "vfx_device", return_value=torch.device("cpu")))
        self.stack.enter_context(patch.object(torch.cuda, "device", lambda d: contextlib.nullcontext()))
        self.stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=types.SimpleNamespace(cuda_stream=0)))
        self.vsr = FakeVSR()
        self.stack.enter_context(patch.object(nodes, "_ARTIFACT_SESSION", self.vsr))
        self.stack.enter_context(patch.object(nodes, "_NATIVE_SESSIONS", {}))
        self.stack.enter_context(patch.object(native, "NativeSession", FakeNative))

    def test_artifact_dimensions_dtype_audio_and_levels(self):
        audio = {"waveform": object(), "sample_rate": 48000}
        for height, width in ((576, 720), (17, 19)):
            for strength, quality in nodes.ARTIFACT_STRENGTHS.items():
                images = torch.rand(2, height, width, 3)
                output, passed_audio = nodes.RTXArtifactReduction().reduce(images, strength, audio=audio)
                self.assertEqual(output.shape, images.shape)
                self.assertEqual(output.dtype, torch.float16)
                self.assertIs(passed_audio, audio)
                self.assertEqual(self.vsr.calls[-1][:3], (quality, width, height))
                torch.testing.assert_close(output, images.half())

    def test_no_size_controls_and_defaults(self):
        with patch.object(nodes, "get_quality_enum", side_effect=ImportError):
            schema = nodes.RTXArtifactReduction.INPUT_TYPES()
        self.assertEqual(set(schema["required"]), {"images", "strength", "output_dtype", "keep_loaded"})
        self.assertEqual(schema["required"]["strength"][1]["default"], "LOW")
        self.assertEqual(schema["required"]["output_dtype"][1]["default"], "float16")
        self.assertTrue(schema["required"]["keep_loaded"][1]["default"])

    def test_strengths_follow_installed_sdk(self):
        with patch.object(nodes, "get_quality_enum", return_value=[types.SimpleNamespace(name="DENOISE_LOW")]):
            self.assertEqual(nodes.get_artifact_strengths(), ["LOW"])

    def test_cleanup_success_failure_and_cancel(self):
        images = torch.rand(1, 11, 13, 3)
        nodes.RTXArtifactReduction().reduce(images, keep_loaded=False)
        self.assertEqual(self.vsr.closed, 1)
        self.vsr.bad_size = True
        with self.assertRaisesRegex(RuntimeError, "dimensions"):
            nodes.RTXArtifactReduction().reduce(images)
        self.assertEqual(self.vsr.closed, 2)
        with patch.object(mm, "throw_exception_if_processing_interrupted", side_effect=RuntimeError("cancel")):
            with self.assertRaisesRegex(RuntimeError, "cancel"):
                nodes.RTXArtifactReduction().reduce(images)
        self.assertEqual(self.vsr.closed, 3)

    def test_allocation_and_load_failure_cleanup(self):
        images = torch.rand(1, 11, 13, 3)
        with patch.object(nodes, "allocate_output", side_effect=MemoryError):
            with self.assertRaises(MemoryError):
                nodes.RTXArtifactReduction().reduce(images)
        with patch.object(self.vsr, "get", side_effect=RuntimeError("load")):
            with self.assertRaisesRegex(RuntimeError, "load"):
                nodes.RTXArtifactReduction().reduce(images)
        self.assertEqual(self.vsr.closed, 2)

    def test_foreground_mask_and_temporal_reset(self):
        images = torch.ones(2, 288, 512, 3)
        fg, mask = nodes.RTXAIGreenScreen().segment(images, temporal=False, output_dtype="float32")
        self.assertEqual(mask.shape, (2, 288, 512))
        self.assertEqual(mask.dtype, torch.float32)
        torch.testing.assert_close(fg, images * mask.unsqueeze(-1))
        self.assertTrue(torch.all(mask[:, :, :256] == 1))
        self.assertTrue(torch.all(mask[:, :, 256:] == 0))
        session = nodes._NATIVE_SESSIONS["GreenScreen"]
        self.assertEqual(session.resets, 2)
        nodes.RTXAIGreenScreen().segment(images)
        self.assertEqual(session.resets, 3)

    def test_blur_mask_broadcast_and_batch(self):
        images = torch.ones(2, 15, 17, 3)
        for mask in (torch.ones(15, 17), torch.ones(1, 15, 17), torch.ones(2, 15, 17)):
            result, = nodes.RTXBackgroundBlur().blur(images, mask)
            torch.testing.assert_close(result, images.half())
        masks = torch.stack([torch.zeros(15, 17), torch.ones(15, 17)])
        result, = nodes.RTXBackgroundBlur().blur(images, masks)
        self.assertEqual(result[0].sum(), 0)
        self.assertEqual(result[1].sum(), 15 * 17 * 3)

    def test_invalid_inputs(self):
        for images in (None, torch.empty(0, 12, 12, 3), torch.empty(1, 12, 12, 2)):
            with self.assertRaises(ValueError):
                nodes.RTXArtifactReduction().reduce(images)
        images = torch.ones(2, 15, 17, 3)
        for mask in (None, torch.ones(3, 15, 17), torch.ones(2, 14, 17)):
            with self.assertRaises(ValueError):
                nodes.RTXBackgroundBlur().blur(images, mask)
        with self.assertRaisesRegex(ValueError, "512x288"):
            nodes.RTXAIGreenScreen().segment(images)
        with self.assertRaises(ValueError):
            nodes.RTXBackgroundBlur().blur(images, torch.ones(15, 17), strength=2)

    def test_rgba_and_optional_audio(self):
        images = torch.rand(1, 13, 17, 4)
        result, audio = nodes.RTXArtifactReduction().reduce(images, output_dtype="float32")
        torch.testing.assert_close(result, images[..., :3])
        self.assertIsNone(audio)

    def test_native_descriptor_abi_and_format(self):
        # 64-bit ABI from nvCVImage.h; descriptor references caller-owned data.
        self.assertEqual(ctypes.sizeof(native.NvCVImage), 64)
        self.assertEqual(native.NvCVImage.pixels.offset, 32)
        tensor = Mock(dtype=torch.uint8, is_cuda=True, ndim=3, shape=(576, 720, 3))
        tensor.is_contiguous.return_value = True
        tensor.stride.return_value = 2160
        tensor.data_ptr.return_value = 123456
        tensor.numel.return_value = 576 * 720 * 3
        desc = native.image_descriptor(tensor)
        self.assertEqual((desc.width, desc.height, desc.pitch), (720, 576, 2160))
        self.assertEqual((desc.pixelFormat, desc.componentType, desc.gpuMem), (5, 1, 1))
        self.assertEqual(desc.pixels, 123456)
        self.assertIsNone(desc.deletePtr)


class NativeSessionTests(unittest.TestCase):
    def test_native_cache_parameters_state_and_cleanup(self):
        calls = []

        def call(name, *args):
            calls.append((name, args))
            if name == "CreateEffect":
                args[1]._obj.value = 100
            elif name == "AllocateState":
                args[1]._obj.value = 200

        api = types.SimpleNamespace(call=call, model_dir=None)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(native, "NativeAPI", return_value=api))
            stack.enter_context(patch.object(native, "image_descriptor", return_value=native.NvCVImage()))
            stack.enter_context(patch.object(torch.cuda, "device", lambda d: contextlib.nullcontext()))
            stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=types.SimpleNamespace(cuda_stream=123)))
            stack.enter_context(patch.object(torch.cuda, "synchronize"))
            session = native.NativeSession("GreenScreen")
            session.get(torch.device("cpu"), 288, 512, 2)
            session.get(torch.device("cpu"), 288, 512, 2)
            self.assertEqual(sum(name == "CreateEffect" for name, _ in calls), 1)
            self.assertTrue(any(name == "SetU32" and args[1:] == (b"Mode", 2) for name, args in calls))
            session.reset()
            self.assertTrue(any(name == "ResetState" for name, _ in calls))
            session.get(torch.device("cpu"), 300, 520, 2)
            self.assertEqual(sum(name == "CreateEffect" for name, _ in calls), 2)
            session.close()
            self.assertEqual(sum(name == "DeallocateState" for name, _ in calls), 2)
            self.assertEqual(sum(name == "DestroyEffect" for name, _ in calls), 2)
            self.assertIsNone(session.src)
            session.close()  # idempotent

    def test_native_load_failure_destroys_partial_effect(self):
        calls = []

        def call(name, *args):
            calls.append(name)
            if name == "CreateEffect":
                args[1]._obj.value = 100
            if name == "Load":
                raise RuntimeError("missing model")

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(native, "NativeAPI", return_value=types.SimpleNamespace(call=call, model_dir=None)))
            stack.enter_context(patch.object(native, "image_descriptor", return_value=native.NvCVImage()))
            stack.enter_context(patch.object(torch.cuda, "device", lambda d: contextlib.nullcontext()))
            stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=types.SimpleNamespace(cuda_stream=123)))
            stack.enter_context(patch.object(torch.cuda, "synchronize"))
            session = native.NativeSession("BackgroundBlur")
            with self.assertRaisesRegex(RuntimeError, "missing model"):
                session.get(torch.device("cpu"), 20, 30, 0.5)
            self.assertIn("DestroyEffect", calls)
            self.assertIsNone(session.key)
            self.assertIsNone(session.dst)


if __name__ == "__main__":
    unittest.main()
