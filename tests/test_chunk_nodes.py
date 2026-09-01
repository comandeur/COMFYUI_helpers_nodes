"""Tests for the chunking nodes.

Run with the python ComfyUI runs on, from anywhere:

    python_embeded\\python.exe custom_nodes\\COMFYUI_helpers_nodes\\tests\\test_chunk_nodes.py

ComfyUI itself is not started: folder_paths is stubbed onto a temp directory so
the file-touching nodes can be exercised without going near a real output/.
"""
import math
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(PACK)
COMFY_ROOT = os.path.dirname(CUSTOM_NODES)
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, CUSTOM_NODES)

SANDBOX = tempfile.mkdtemp(prefix="cmdr_chunk_tests_")
OUTPUT_DIR = os.path.join(SANDBOX, "output")
TEMP_DIR = os.path.join(SANDBOX, "temp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_output_directory = lambda: OUTPUT_DIR
folder_paths.get_temp_directory = lambda: TEMP_DIR
folder_paths.get_input_directory = lambda: SANDBOX
sys.modules["folder_paths"] = folder_paths
sys.modules.setdefault("nodes", types.ModuleType("nodes"))

import torch  # noqa: E402

from COMFYUI_helpers_nodes.helpers_nodes.chunk_nodes import (  # noqa: E402
    FrameStorePrune, FrameStoreStatus, H3ChunkPlanner, SaveFrameSequence,
    plan_chunk, resolve_progress, snap_length)

PASSED = []


def check(name):
    PASSED.append(name)
    print(f"  ok  {name}")


# --------------------------------------------------------------------------
# 1. planner: the kept ranges must tile range(T) exactly
# --------------------------------------------------------------------------
def test_coverage_invariant():
    cases = 0
    for clip_seconds in (5, 6, 7, 8, 10, 15):
        length = snap_length(clip_seconds, 24)
        for total in range(100, 3001):
            if total < length:
                continue
            covered = []
            plan = plan_chunk(total, length, 0)
            for k in range(plan["chunk_count"]):
                p = plan_chunk(total, length, k)
                start = p["skip_frames"] + p["trim_start"]
                covered.extend(range(start, start + p["trim_length"]))
            assert covered == list(range(total)), (
                f"clip_seconds={clip_seconds} L={length} T={total}: "
                f"covered {len(covered)} frames, first mismatch around "
                f"{next((i for i, v in enumerate(covered) if v != i), len(covered))}"
            )
            cases += 1
    check(f"coverage invariant over {cases} (T, clip_seconds) combinations")


# --------------------------------------------------------------------------
# 2. planner: the 17k+5 grid, and the trained range
# --------------------------------------------------------------------------
def test_length_grid():
    for clip_seconds in [0.2, 1, 3, 5, 6, 6.58, 7, 8, 10, 15, 20]:
        for fps in (24, 30, 16):
            length = snap_length(clip_seconds, fps)
            assert (length - 5) % 17 == 0, (clip_seconds, fps, length)
            assert 124 <= length <= 362, (clip_seconds, fps, length)
            raw = snap_length(clip_seconds, fps, clamp_to_trained_range=False)
            assert (raw - 5) % 17 == 0, (clip_seconds, fps, raw)
            assert raw >= round(clip_seconds * fps), "must round up, never down"
    assert snap_length(6.0, 24) == 158, snap_length(6.0, 24)
    check("length stays on the 17k+5 grid, rounds up, clamps to [124, 362]")


def test_planner_node():
    node = H3ChunkPlanner()
    (length, skip, trim_start, trim_length, count, is_last, info,
     total_frames) = node.plan(6.0, 24, 3, 1440)
    assert (length, skip, trim_start, trim_length) == (158, 474, 0, 158)
    assert count == 10 and is_last is False and total_frames == 1440
    assert "chunk 3/10" in info and "L=158" in info, info

    assert node.plan(6.0, 24, 9, 1440)[5] is True

    # video_info replaces the typed frame count, and total_frames reports it
    out = node.plan(6.0, 24, 0, 0, video_info={"source_duration": 60.0})
    assert out[4] == math.ceil(1440 / 158), out
    assert out[7] == 1440, out

    # asking past the end is clamped rather than producing a negative length
    plan = node.plan(6.0, 24, 99, 1440)
    assert plan[3] > 0 and plan[5] is True and "past the last chunk" in plan[6]

    for bad in ((6.0, 24, 0, 0), (20.0, 24, 0, 100)):
        try:
            node.plan(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected a refusal for {bad}")
    check("planner node: outputs, video_info, clamping, refusals")


# --------------------------------------------------------------------------
# 3 & 4. frame store status
# --------------------------------------------------------------------------
PNG_1PX = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
           b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
           b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82")


def write_frames(directory, count, prefix="frame", start=0, truncated=()):
    os.makedirs(directory, exist_ok=True)
    for i in range(start, start + count):
        data = PNG_1PX[:20] if i in truncated else PNG_1PX
        with open(os.path.join(directory, f"{prefix}_{i:06d}.png"), "wb") as f:
            f.write(data)


def test_status_is_not_cached():
    assert math.isnan(FrameStoreStatus.IS_CHANGED()), (
        "IS_CHANGED must be nan, or a queue of runs all read the same chunks_done")
    store = os.path.join(OUTPUT_DIR, "status_cache")
    write_frames(store, 8)
    node = FrameStoreStatus()
    first = node.status("status_cache", "frame", 8)
    write_frames(store, 8, start=8)
    second = node.status("status_cache", "frame", 8)
    assert first[0] == 8 and second[0] == 16, (first, second)
    assert first[1] == 1 and second[1] == 2, (first, second)
    check("status re-reads the folder every call (IS_CHANGED is nan)")


def test_status_missing_directory():
    node = FrameStoreStatus()
    count, done, clean, report, complete = node.status("never_written", "frame", 8)
    assert (count, done, clean, complete) == (0, 0, True, False)
    assert "does not exist yet" in report
    check("a missing folder is a starting state, not an error")


def test_status_auto_repair():
    store = os.path.join(OUTPUT_DIR, "status_repair")
    write_frames(store, 21, truncated=(20,))
    node = FrameStoreStatus()

    count, done, clean, report, complete = node.status("status_repair", "frame", 8)
    assert (count, done, clean, complete) == (20, 2, False, False)
    assert "incomplete" in report

    count, done, clean, report, complete = node.status("status_repair", "frame", 8,
                                                       auto_repair=True)
    assert (count, done, clean, complete) == (16, 2, True, False)
    assert len(os.listdir(store)) == 16, os.listdir(store)
    check("auto_repair drops the truncated file and the partial chunk's tail")


# --------------------------------------------------------------------------
# the completion rule, and what it protects
# --------------------------------------------------------------------------
def test_progress_rule():
    # a finished 1440-frame job in chunks of 158 holds 1440 frames, not 10*158
    assert resolve_progress(1440, 158, 1440) == (10, True, True)
    # without the target, the same store reads as unfinished -- this is the
    # v0.6.0 behaviour, kept for anyone who leaves total_frames at 0
    assert resolve_progress(1440, 158, 0) == (9, False, False)
    # mid-series
    assert resolve_progress(1422, 158, 1440) == (9, False, True)
    assert resolve_progress(0, 158, 1440) == (0, False, True)
    # overfilled by an earlier series: complete, but not clean
    assert resolve_progress(1458, 158, 1440) == (10, True, False)
    # corruption and holes keep it dirty either side of the rule
    assert resolve_progress(1440, 158, 1440, has_corrupt=True)[2] is False
    assert resolve_progress(1422, 158, 1440, has_gaps=True)[2] is False
    check("progress rule: a short last chunk still counts as a whole chunk")


def test_auto_repair_spares_a_finished_store():
    store = os.path.join(OUTPUT_DIR, "repair_complete")
    write_frames(store, 1440)
    node = FrameStoreStatus()

    count, done, clean, report, complete = node.status(
        "repair_complete", "frame", 158, verify_integrity=False,
        auto_repair=True, total_frames=1440)
    assert (count, done, clean, complete) == (1440, 10, True, True)
    assert len(os.listdir(store)) == 1440, "a complete store must not be touched"

    # and with no target it still trims, which is exactly why total_frames
    # should always be wired
    count, done, clean, report, complete = node.status(
        "repair_complete", "frame", 158, verify_integrity=False, auto_repair=True)
    assert (count, done, clean, complete) == (1422, 9, True, False)
    shutil.rmtree(store)
    check("auto_repair leaves a finished store alone when it knows the target")


def test_overfilled_store_is_reported():
    store = os.path.join(OUTPUT_DIR, "overfilled")
    write_frames(store, 1458)
    count, done, clean, report, complete = FrameStoreStatus().status(
        "overfilled", "frame", 158, verify_integrity=False, total_frames=1440)
    assert (count, done, complete) == (1458, 10, True)
    assert clean is False, "duplicates from an earlier series are not clean"
    assert "18 frame(s) past the target" in report, report
    shutil.rmtree(store)
    check("frames past the target are reported, never silently accepted")


def run_resume_loop(directory, total, clip_seconds, extra_runs=3, fps=24):
    """Drive Status -> Planner -> Save until the series is done, then keep going."""
    planner, status, saver = H3ChunkPlanner(), FrameStoreStatus(), SaveFrameSequence()
    length = snap_length(clip_seconds, fps)
    chunk_count = -(-total // length)
    writes = []
    for _ in range(chunk_count + extra_runs):
        _, chunks_done, _, _, _ = status.status(directory, "frame", length,
                                                verify_integrity=False,
                                                total_frames=total)
        plan = planner.plan(clip_seconds, fps, chunks_done, total)
        trim_length, planned_total = plan[3], plan[7]
        images = torch.zeros((trim_length, 2, 2, 3))
        written, _, _ = saver.save(images, directory, "frame", "png", False,
                                   stop_at_frames=planned_total)
        writes.append(written)
    final = status.status(directory, "frame", length, verify_integrity=False,
                          total_frames=total)
    return writes, final, chunk_count


def test_resume_loop_converges_on_disk():
    for total, clip_seconds in ((200, 6), (400, 15), (611, 5), (1440, 6), (900, 8)):
        directory = f"loop_{total}_{clip_seconds}"
        writes, final, chunk_count = run_resume_loop(directory, total, clip_seconds)
        count, chunks_done, clean, report, complete = final
        assert count == total, (total, clip_seconds, count)
        assert complete is True and clean is True, (total, clip_seconds, report)
        assert chunks_done == chunk_count, (total, clip_seconds, chunks_done, chunk_count)
        assert sum(writes) == total, (total, clip_seconds, writes)
        assert writes[-3:] == [0, 0, 0], (total, clip_seconds, writes)
        shutil.rmtree(os.path.join(OUTPUT_DIR, directory))
    check("resume loop on real files: converges to exactly T, extra runs write nothing")


def test_resume_loop_matrix():
    """The same loop over the wide matrix, without touching the disk.

    Writing every frame of 126 combinations would take about ten minutes on
    Windows, so the file-backed version above covers a handful of shapes and
    this one covers the arithmetic everywhere, through the same
    resolve_progress and plan_chunk the nodes call.
    """
    combinations = 0
    for clip_seconds in (5, 6, 7, 8, 10, 15):
        length = snap_length(clip_seconds, 24)
        for total in range(200, 3001, 137):
            if total < length:
                continue
            chunk_count = -(-total // length)
            stored = 0
            for _ in range(chunk_count + 3):
                chunks_done, complete, _ = resolve_progress(stored, length, total)
                plan = plan_chunk(total, length, chunks_done)
                if stored >= total:          # SaveFrameSequence's stop_at_frames
                    continue
                stored += plan["trim_length"]
            chunks_done, complete, clean = resolve_progress(stored, length, total)
            assert stored == total, (clip_seconds, total, stored)
            assert complete and clean and chunks_done == chunk_count
            combinations += 1
    check(f"resume loop converges exactly on {combinations} (T, clip_seconds) combinations")


# --------------------------------------------------------------------------
# 5, 6, 7, 8. prune
# --------------------------------------------------------------------------
def test_prune_refusals():
    store = os.path.join(OUTPUT_DIR, "prune_refuse")
    write_frames(store, 10)
    node = FrameStorePrune()

    deleted, _, report = node.prune("prune_refuse", "frame", "disabled", 0, True)
    assert deleted == 0 and "disabled" in report

    deleted, _, report = node.prune("prune_refuse", "frame", "after_assembly", 0, True)
    assert deleted == 0 and "no filenames" in report, report

    empty = os.path.join(OUTPUT_DIR, "empty_final.mp4")
    open(empty, "wb").close()
    deleted, _, report = node.prune("prune_refuse", "frame", "after_assembly", 0, True,
                                    filenames=(True, [empty]))
    assert deleted == 0 and "missing or empty" in report, report

    good = os.path.join(OUTPUT_DIR, "good_final.mp4")
    with open(good, "wb") as f:
        f.write(b"x" * 32)
    deleted, _, report = node.prune("prune_refuse", "frame", "after_assembly", 999, True,
                                    filenames=(True, [good]))
    assert deleted == 0 and "expected 999" in report, report
    assert len(os.listdir(store)) == 10, "nothing may be deleted on a refusal"

    os.remove(empty)
    os.remove(good)
    shutil.rmtree(store)
    check("prune refuses: disabled, no filenames, empty final, wrong frame count")


def test_prune_path_escape():
    node = FrameStorePrune()
    for bad in ("../../etc", "..", "/etc", "C:\\Windows", ""):
        try:
            node.prune(bad, "frame", "always", 0, True)
        except ValueError:
            continue
        raise AssertionError(f"escaping directory accepted: {bad!r}")
    check("prune refuses to leave output/")


def test_prune_intermediates():
    store = os.path.join(OUTPUT_DIR, "prune_inter")
    write_frames(store, 4)
    names = ["a.png", "b.mp4", "c-audio.mp4"]
    paths = []
    for name in names:
        p = os.path.join(OUTPUT_DIR, name)
        with open(p, "wb") as f:
            f.write(b"x" * 64)
        paths.append(p)
    outsider = os.path.join(SANDBOX, "not_ours.mp4")
    with open(outsider, "wb") as f:
        f.write(b"x" * 8)

    node = FrameStorePrune()
    deleted, final_video, report = node.prune(
        "prune_inter", "frame", "after_assembly", 4, True,
        filenames=(True, [outsider] + paths))

    assert final_video == paths[-1], final_video
    assert os.path.isfile(paths[-1]), "the final video must survive"
    assert not os.path.exists(paths[0]) and not os.path.exists(paths[1])
    assert os.path.isfile(outsider), "a path outside output/ must be left alone"
    assert "outside output/ and temp/" in report
    assert deleted == 4 + 2, deleted
    assert not os.path.exists(store), "the emptied folder should be gone"

    os.remove(paths[-1])
    os.remove(outsider)
    check("prune keeps only the final video and never touches paths outside output/")


def test_full_cycle_leaves_only_the_video():
    for name in os.listdir(OUTPUT_DIR):
        target = os.path.join(OUTPUT_DIR, name)
        shutil.rmtree(target) if os.path.isdir(target) else os.remove(target)

    images = torch.rand((6, 8, 12, 3))
    saver = SaveFrameSequence()
    written, next_index, path = saver.save(images, "cycle", "frame", "png", False)
    assert (written, next_index) == (6, 6)
    written, next_index, _ = saver.save(images, "cycle", "frame", "png", False)
    assert (written, next_index) == (6, 12), "a second run must append, not overwrite"

    status = FrameStoreStatus()
    count, done, clean, _, complete = status.status("cycle", "frame", 6)
    assert (count, done, clean, complete) == (12, 2, True, False)

    final = os.path.join(OUTPUT_DIR, "FINAL_00001-audio.mp4")
    silent = os.path.join(OUTPUT_DIR, "FINAL_00001.mp4")
    for p in (final, silent):
        with open(p, "wb") as f:
            f.write(b"x" * 128)

    deleted, final_video, _ = FrameStorePrune().prune(
        "cycle", "frame", "after_assembly", count, True,
        filenames=(True, [silent, final]))
    assert deleted == 13 and final_video == final, (deleted, final_video)

    left = [os.path.join(r, f) for r, _, fs in os.walk(OUTPUT_DIR) for f in fs]
    assert left == [final], left
    check("after a full cycle, output/ holds nothing but the final video")


def test_save_formats_and_start_index():
    node = SaveFrameSequence()
    images = torch.rand((3, 8, 12, 3))
    for fmt, extension in (("png", ".png"), ("webp_lossless", ".webp"),
                           ("webp_q95", ".webp"), ("jpeg_q95", ".jpg")):
        written, next_index, path = node.save(images, f"fmt_{fmt}", "frame", fmt, False)
        assert written == 3 and next_index == 3
        names = sorted(os.listdir(path))
        assert names == [f"frame_{i:06d}{extension}" for i in range(3)], names
        assert not any(n.endswith(".tmp") for n in names), "no temp file may survive"

    written, next_index, path = node.save(images, "fmt_start", "frame", "png", False,
                                          start_index=100)
    assert next_index == 103
    assert sorted(os.listdir(path))[0] == "frame_000100.png"

    written, next_index, _ = node.save(images, "fmt_start", "frame", "png", False)
    assert next_index == 106, "-1 continues after the highest existing index"

    # six digits, so a long shoot doesn't wrap at 99999
    node.save(images, "fmt_wide", "frame", "png", False, start_index=999998)
    assert "frame_999998.png" in os.listdir(os.path.join(OUTPUT_DIR, "fmt_wide"))
    assert "frame_1000000.png" in os.listdir(os.path.join(OUTPUT_DIR, "fmt_wide"))
    check("save: every format, atomic writes, start_index and 6-digit numbering")


def test_stop_at_frames():
    node = SaveFrameSequence()
    images = torch.rand((4, 8, 12, 3))
    written, next_index, _ = node.save(images, "stop", "frame", "png", False,
                                       stop_at_frames=10)
    assert (written, next_index) == (4, 4)
    written, next_index, _ = node.save(images, "stop", "frame", "png", False,
                                       stop_at_frames=10)
    assert (written, next_index) == (4, 8)
    # the third run would cross the limit, so it writes nothing at all
    written, next_index, path = node.save(images, "stop", "frame", "png", False,
                                          stop_at_frames=8)
    assert (written, next_index) == (0, 8), (written, next_index)
    assert len(os.listdir(path)) == 8
    # 0 keeps the old, unlimited behaviour
    written, _, _ = node.save(images, "stop", "frame", "png", False, stop_at_frames=0)
    assert written == 4
    check("stop_at_frames turns a queue that ran too long into a no-op")


if __name__ == "__main__":
    print(f"sandbox: {SANDBOX}")
    try:
        test_coverage_invariant()
        test_length_grid()
        test_planner_node()
        test_status_is_not_cached()
        test_status_missing_directory()
        test_status_auto_repair()
        test_progress_rule()
        test_auto_repair_spares_a_finished_store()
        test_overfilled_store_is_reported()
        test_resume_loop_matrix()
        test_resume_loop_converges_on_disk()
        test_save_formats_and_start_index()
        test_stop_at_frames()
        test_prune_refusals()
        test_prune_path_escape()
        test_prune_intermediates()
        test_full_cycle_leaves_only_the_video()
        print(f"\n{len(PASSED)} checks passed")
    finally:
        shutil.rmtree(SANDBOX, ignore_errors=True)
