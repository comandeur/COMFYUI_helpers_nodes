"""Tests for the resolution nodes.

    python_embeded\\python.exe custom_nodes\\COMFYUI_helpers_nodes\\tests\\test_resolution_nodes.py
"""
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(PACK)
sys.path.insert(0, CUSTOM_NODES)
sys.modules.setdefault("nodes", types.ModuleType("nodes"))

from COMFYUI_helpers_nodes.helpers_nodes.resolution_nodes import (  # noqa: E402
    ASPECT_RATIOS, ResolutionSelector, ScaleResolutionToMegapixels,
    scale_to_megapixels)

PASSED = []


def check(name):
    PASSED.append(name)
    print(f"  ok  {name}")


def test_presets_cover_the_picker():
    expected = ["1:1 (Square)",
                "5:6 (Portrait Near-Square)", "6:5 (Near-Square)",
                "3:4 (Portrait Standard)", "4:3 (Standard)",
                "2:3 (Portrait Photo)", "3:2 (Photo)",
                "9:16 (Portrait Widescreen)", "16:9 (Widescreen)",
                "9:21 (Portrait Ultrawide)", "21:9 (Ultrawide)"]
    assert list(ASPECT_RATIOS) == expected, list(ASPECT_RATIOS)
    # every preset core offers must still be here, spelled identically
    for name in ("1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)",
                 "3:4 (Portrait Standard)", "4:3 (Standard)",
                 "9:16 (Portrait Widescreen)", "16:9 (Widescreen)",
                 "21:9 (Ultrawide)"):
        assert name in ASPECT_RATIOS, name
    check("eleven presets: core's eight, plus 9:21, 5:6 and 6:5")


def test_matches_core_when_asked():
    """With base 1024x1024 the target area is core's, so it is a drop-in."""
    node = ResolutionSelector()
    for name, (rw, rh) in ASPECT_RATIOS.items():
        for megapixels in (0.3, 0.5, 0.9, 1.0, 2.0):
            for multiple in (8, 32):
                scale = math.sqrt(megapixels * 1024 * 1024 / (rw * rh))
                core = (round(rw * scale / multiple) * multiple,
                        round(rh * scale / multiple) * multiple)
                mine = node.select(name, megapixels, multiple,
                                   megapixel_base="1024x1024")
                area_gap = abs(mine[0] * mine[1] - core[0] * core[1]) / (core[0] * core[1])
                mine_ratio = abs(mine[0] / mine[1] - rw / rh) / (rw / rh)
                core_ratio = abs(core[0] / core[1] - rw / rh) / (rw / rh)
                # same ballpark as core, and never a worse aspect ratio
                assert area_gap < 0.12, (name, megapixels, multiple, core, mine)
                assert mine_ratio <= core_ratio + 1e-9, (name, megapixels, multiple,
                                                         core, mine)
    check("base 1024x1024 tracks core's area, with a ratio at least as accurate")


def test_selector_accuracy():
    node = ResolutionSelector()
    worst_ratio = 0.0
    worst_pixels = 0.0
    for name, (rw, rh) in ASPECT_RATIOS.items():
        wanted = rw / rh
        for megapixels in (0.3, 0.31, 0.99, 1.0, 1.5, 2.5, 4.0):
            for multiple in (1, 8, 16, 32, 64):
                width, height = node.select(name, megapixels, multiple)
                assert width % multiple == 0 and height % multiple == 0
                assert width > 0 and height > 0
                assert (width > height) == (wanted > 1) or wanted == 1
                ratio_error = abs(width / height - wanted) / wanted
                pixel_error = abs(width * height / 1e6 - megapixels) / megapixels
                worst_ratio = max(worst_ratio, ratio_error)
                worst_pixels = max(worst_pixels, pixel_error)
                # a coarse grid on a small budget is the hard case; stay sane
                assert ratio_error < 0.15, (name, megapixels, multiple, width, height)
    check(f"every preset x budget x multiple: ratio off by at most "
          f"{worst_ratio * 100:.1f}%, pixels by {worst_pixels * 100:.1f}%")


def test_exact_hits():
    node = ResolutionSelector()
    assert node.select("1:1 (Square)", 1.0, 8) == (1000, 1000)
    assert node.select("2:3 (Portrait Photo)", 0.3, 32) == (448, 672)
    assert node.select("16:9 (Widescreen)", 2.0, 8) == (1880, 1056)
    assert node.select("21:9 (Ultrawide)", 1.0, 32) == (1504, 640)
    # exact ratios stay exact when the grid allows it
    for name, (rw, rh) in ASPECT_RATIOS.items():
        width, height = node.select(name, 1.0, 1)
        assert abs(width / height - rw / rh) / (rw / rh) < 0.002, (name, width, height)
    check("known resolutions come out exactly, ratios hold on a free grid")


def test_megapixel_step_is_fine_grained():
    for cls in (ResolutionSelector, ScaleResolutionToMegapixels):
        spec = cls.INPUT_TYPES()["required"]["megapixels"][1]
        assert spec["step"] == 0.01, (cls.__name__, spec)
        assert spec["min"] <= 0.01
    node = ResolutionSelector()
    # on a free grid every 0.01 step must land somewhere new
    fine = {node.select("3:2 (Photo)", round(0.30 + i * 0.01, 2), 1) for i in range(6)}
    assert len(fine) == 6, fine
    # on a multiple of 8 the grid itself quantises, so steps may collide -- but
    # the range still has to move, which a step of 0.1 could not do here
    coarse = {node.select("3:2 (Photo)", round(0.30 + i * 0.01, 2), 8) for i in range(6)}
    assert len(coarse) >= 3, coarse
    check("megapixels moves in steps of 0.01, and the result follows")


def test_pair_node_still_agrees():
    node = ScaleResolutionToMegapixels()
    for name, (rw, rh) in ASPECT_RATIOS.items():
        for megapixels in (0.3, 1.0, 2.0):
            for multiple in (8, 32):
                assert (node.scale(rw, rh, megapixels, multiple)
                        == scale_to_megapixels(rw, rh, megapixels, multiple)
                        == ResolutionSelector().select(name, megapixels, multiple))
    check("both nodes share one implementation and agree on every preset")


if __name__ == "__main__":
    test_presets_cover_the_picker()
    test_matches_core_when_asked()
    test_selector_accuracy()
    test_exact_hits()
    test_megapixel_step_is_fine_grained()
    test_pair_node_still_agrees()
    print(f"\n{len(PASSED)} checks passed")
