"""Chunked long-video processing: plan the cuts, track what's on disk, clean up.

Built for the MiniMax H3 inpainting pipeline, where a long source is processed a
clip at a time across several runs and the frames live in an output folder
between them. Nothing here is H3-specific except the 17k+5 length grid.

The four nodes cover one loop:

    plan the chunk  ->  save its frames  ->  read back where we are  ->  prune

``FrameStoreStatus`` is what makes the loop resumable: it reads the folder and
says which chunk comes next, so re-queueing the workflow picks up where it
stopped instead of needing a hand-managed index.
"""
import json
import os
import re

import numpy as np
from PIL import Image, PngImagePlugin

import folder_paths
from comfy.utils import ProgressBar

# MiniMax H3 accepts clip lengths on a 17k+5 grid, and was trained between these
# two bounds. Both ends sit on the grid already (124 = 17*7+5, 362 = 17*21+5).
LENGTH_STEP = 17
LENGTH_OFFSET = 5
TRAINED_MIN = 124
TRAINED_MAX = 362

IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")
INDEX_PATTERN = re.compile(r"(\d+)\D*$")

SEQUENCE_FORMATS = {
    "png": {"extension": ".png", "pil_format": "PNG", "options": {"compress_level": 4}},
    "webp_lossless": {"extension": ".webp", "pil_format": "WEBP",
                      "options": {"lossless": True, "quality": 100}},
    "webp_q95": {"extension": ".webp", "pil_format": "WEBP",
                 "options": {"lossless": False, "quality": 95}},
    "jpeg_q95": {"extension": ".jpg", "pil_format": "JPEG",
                 "options": {"quality": 95, "subsampling": 0}},
}

PRUNE_MODES = ["disabled", "after_assembly", "always"]


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def resolve_store_directory(directory, must_exist=False):
    """Resolve ``directory`` under ComfyUI's output folder, refusing escapes.

    Same guard VHS_PruneOutputs applies before deleting anything: a workflow
    someone else wrote must not be able to reach outside output/ through '..'
    or an absolute path.
    """
    directory = (directory or "").strip().strip('"')
    if not directory:
        raise ValueError("directory is empty: give a folder name under ComfyUI's output/")
    if os.path.isabs(directory) or os.path.splitdrive(directory)[0]:
        raise ValueError(f"directory must be relative to output/, got {directory!r}")

    base = os.path.abspath(folder_paths.get_output_directory())
    path = os.path.abspath(os.path.join(base, directory))
    try:
        inside = os.path.commonpath([base, path]) == base
    except ValueError:  # different drives on Windows
        inside = False
    if not inside or path == base:
        raise ValueError(f"directory must stay inside output/ and name a subfolder, got {directory!r}")
    if must_exist and not os.path.isdir(path):
        raise ValueError(f"directory does not exist: {path}")
    return path


def frame_index(filename):
    """The trailing number in a filename, or None. Sorts like VHS_LoadImagesPath."""
    match = INDEX_PATTERN.search(os.path.splitext(filename)[0])
    return int(match.group(1)) if match else None


def list_frame_files(path, prefix):
    """[(index, filename)] for ``prefix*`` image files, ordered numerically."""
    if not os.path.isdir(path):
        return []
    found = []
    for name in os.listdir(path):
        if not name.startswith(prefix):
            continue
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        if not os.path.isfile(os.path.join(path, name)):
            continue
        index = frame_index(name)
        if index is not None:
            found.append((index, name))
    found.sort()
    return found


def is_file_complete(path):
    """Whether the file carries its own end marker, i.e. the write finished."""
    try:
        size = os.path.getsize(path)
        if size < 12:
            return False
        with open(path, "rb") as f:
            head = f.read(12)
            f.seek(-12, os.SEEK_END)
            tail = f.read(12)
    except OSError:
        return False

    extension = os.path.splitext(path)[1].lower()
    if extension == ".png":
        return tail[-8:-4] == b"IEND"
    if extension == ".webp":
        if head[:4] != b"RIFF" or head[8:12] != b"WEBP":
            return False
        return int.from_bytes(head[4:8], "little") + 8 <= size
    if extension in (".jpg", ".jpeg"):
        return tail[-2:] == b"\xff\xd9"
    return True


# --------------------------------------------------------------------------
# A. chunk planner
# --------------------------------------------------------------------------

def snap_length(clip_seconds, fps, clamp_to_trained_range=True):
    """Clip length rounded *up* onto the 17k+5 grid, optionally clamped."""
    length = max(LENGTH_OFFSET, round(clip_seconds * fps))
    length += (LENGTH_OFFSET - (length % LENGTH_STEP)) % LENGTH_STEP
    if clamp_to_trained_range:
        length = min(max(length, TRAINED_MIN), TRAINED_MAX)
    return int(length)


def plan_chunk(source_frames, length, chunk_index):
    """Where chunk ``k`` starts, and what to keep of it.

    The last chunk's window is pulled back so it ends exactly on the source's
    last frame; the overlap it then has with the previous chunk is trimmed off
    the front of what we keep. Concatenating the kept ranges over every k
    reproduces range(T) exactly -- no gap, no duplicate.
    """
    total = int(source_frames)
    if total <= 0:
        raise ValueError("source_frames is 0: connect video_info, or type the frame count")
    if total < length:
        raise ValueError(
            f"clip_seconds is longer than the source: the window is {length} frames "
            f"but the source only has {total}. Lower clip_seconds."
        )

    chunk_count = -(-total // length)  # ceil
    index = min(max(int(chunk_index), 0), chunk_count - 1)

    skip = min(index * length, max(0, total - length))
    trim_start = max(0, (index + 1) * length - total)
    return {
        "length": length,
        "skip_frames": skip,
        "trim_start": trim_start,
        "trim_length": length - trim_start,
        "chunk_count": chunk_count,
        "chunk_index": index,
        "clamped": index != int(chunk_index),
        "is_last": index >= chunk_count - 1,
    }


class H3ChunkPlanner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_seconds": ("FLOAT", {"default": 6.0, "min": 0.2, "max": 20.0, "step": 0.1,
                                           "tooltip": "Wanted clip duration. The real length is "
                                                      "rounded up onto H3's 17k+5 grid."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000,
                                        "control_after_generate": True,
                                        "tooltip": "Which chunk to plan. Feed chunks_done from "
                                                   "Frame Store Status to make runs resumable."}),
                "source_frames": ("INT", {"default": 0, "min": 0, "max": 10 ** 7,
                                          "tooltip": "Total frames of the source. Ignored when "
                                                     "video_info is connected."}),
            },
            "optional": {
                "video_info": ("VHS_VIDEOINFO", {"tooltip": "source_duration x fps, so the frame "
                                                            "count doesn't have to be typed."}),
                "clamp_to_trained_range": ("BOOLEAN", {
                    "default": True,
                    "tooltip": f"Keep the length within H3's trained range "
                               f"[{TRAINED_MIN}, {TRAINED_MAX}].",
                }),
            },
        }

    CATEGORY = "Helpers 🧰/h3"

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("length", "skip_frames", "trim_start", "trim_length",
                    "chunk_count", "is_last", "info")

    FUNCTION = "plan"

    DESCRIPTION = ("Cut a long source into MiniMax H3 sized clips: length on the 17k+5 grid, "
                   "the last window pulled back to land on the final frame.")

    def plan(self, clip_seconds, fps, chunk_index, source_frames,
             video_info=None, clamp_to_trained_range=True):
        if video_info is not None:
            duration = float(video_info.get("source_duration") or 0.0)
            source_frames = int(round(duration * fps))

        wanted = snap_length(clip_seconds, fps, clamp_to_trained_range=False)
        length = snap_length(clip_seconds, fps, clamp_to_trained_range)
        plan = plan_chunk(source_frames, length, chunk_index)

        start = plan["skip_frames"] + plan["trim_start"]
        notes = []
        if clamp_to_trained_range and wanted != length:
            notes.append(f"length clamped from {wanted} to {length} "
                         f"(trained range {TRAINED_MIN}-{TRAINED_MAX})")
        if plan["clamped"]:
            notes.append(f"chunk_index {chunk_index} is past the last chunk, "
                         f"planned {plan['chunk_index']} instead")
        info = (f"chunk {plan['chunk_index']}/{plan['chunk_count']} · L={length} "
                f"({length / fps:.2f}s) · frames {start}→{start + plan['trim_length']} "
                f"· keep {plan['trim_length']}")
        if notes:
            info += "\n" + "\n".join(notes)

        return (plan["length"], plan["skip_frames"], plan["trim_start"], plan["trim_length"],
                plan["chunk_count"], plan["is_last"], info)


# --------------------------------------------------------------------------
# B. frame store status
# --------------------------------------------------------------------------

def read_frame_store(path, prefix, length, verify_integrity=True, auto_repair=False):
    """Inspect a frame folder and work out where to resume."""
    entries = list_frame_files(path, prefix)
    corrupt, valid = [], []
    for index, name in entries:
        full = os.path.join(path, name)
        if verify_integrity and not is_file_complete(full):
            corrupt.append((index, name))
        else:
            valid.append((index, name))

    removed = []
    if auto_repair and corrupt:
        for index, name in corrupt:
            try:
                os.remove(os.path.join(path, name))
                removed.append(name)
            except OSError:
                pass
        corrupt = []

    # a gap means a frame went missing in the middle; the run is not clean and
    # only the contiguous head can be trusted as "done"
    gaps = [b[0] for a, b in zip(valid, valid[1:]) if b[0] != a[0] + 1]

    if auto_repair and valid and len(valid) % length != 0:
        keep = (len(valid) // length) * length
        for index, name in valid[keep:]:
            try:
                os.remove(os.path.join(path, name))
                removed.append(name)
            except OSError:
                pass
        valid = valid[:keep]

    frame_count = len(valid)
    chunks_done = frame_count // length
    is_clean = frame_count % length == 0 and not corrupt and not gaps

    report = [f"{frame_count} frames in {path or '(missing)'}",
              f"chunks_done = {frame_count} // {length} = {chunks_done}"]
    if corrupt:
        report.append(f"{len(corrupt)} incomplete file(s): "
                      + ", ".join(name for _, name in corrupt[:5])
                      + (" ..." if len(corrupt) > 5 else ""))
    if gaps:
        report.append(f"gap(s) in the numbering before index {gaps[:5]}")
    if removed:
        report.append(f"auto_repair removed {len(removed)} file(s)")
    if frame_count % length:
        report.append(f"{frame_count % length} frame(s) past the last whole chunk"
                      + ("" if auto_repair else " (auto_repair would drop them)"))
    if is_clean:
        report.append("clean: resume at chunk " + str(chunks_done))

    return {
        "frame_count": frame_count,
        "chunks_done": chunks_done,
        "is_clean": is_clean,
        "report": "\n".join(report),
    }


class FrameStoreStatus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "H3_inpaint",
                                         "tooltip": "Folder under ComfyUI's output/."}),
                "prefix": ("STRING", {"default": "frame"}),
                "length": ("INT", {"default": 158, "min": 1, "max": 10 ** 7,
                                   "tooltip": "Frames per chunk, to turn a frame count into "
                                              "a chunk count. Feed it the planner's length."}),
            },
            "optional": {
                "verify_integrity": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check each file's end marker, so a run interrupted mid-write "
                               "is not counted as done.",
                }),
                "auto_repair": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Delete incomplete files and the tail of a partial chunk, to "
                               "land back on a whole multiple of length.",
                }),
            },
        }

    CATEGORY = "Helpers 🧰/io"

    RETURN_TYPES = ("INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("frame_count", "chunks_done", "is_clean", "report")

    FUNCTION = "status"

    DESCRIPTION = ("Read a frame folder and say which chunk comes next, so a queued run "
                   "resumes on its own.")

    def status(self, directory, prefix, length, verify_integrity=True, auto_repair=False):
        try:
            path = resolve_store_directory(directory)
        except ValueError as e:
            raise ValueError(str(e)) from e
        if not os.path.isdir(path):
            # nothing written yet is a normal starting state, not an error
            return (0, 0, True, f"{path} does not exist yet: starting at chunk 0")
        result = read_frame_store(path, prefix, length, verify_integrity, auto_repair)
        return (result["frame_count"], result["chunks_done"],
                result["is_clean"], result["report"])

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # The whole point of this node is reading state that changes between
        # runs of the same queue. Caching it would hand every run the same
        # chunks_done and the resume would silently stop advancing.
        return float("nan")


# --------------------------------------------------------------------------
# C. frame store prune
# --------------------------------------------------------------------------

def is_under_comfy_dirs(path):
    """True when the path sits inside output/ or temp/ -- nothing else is ours."""
    target = os.path.abspath(path)
    for base in (folder_paths.get_output_directory(), folder_paths.get_temp_directory()):
        base = os.path.abspath(base)
        try:
            if os.path.commonpath([base, target]) == base:
                return True
        except ValueError:
            continue
    return False


def split_filenames(filenames):
    """VHS_FILENAMES is ``(save_output, [paths...])``, final file last."""
    if not filenames:
        return None, []
    paths = filenames[1] if isinstance(filenames, (list, tuple)) and len(filenames) > 1 else []
    paths = [p for p in (paths or []) if isinstance(p, str)]
    if not paths:
        return None, []
    return paths[-1], paths[:-1]


class FrameStorePrune:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "H3_inpaint"}),
                "prefix": ("STRING", {"default": "frame",
                                      "tooltip": "Only files starting with this are removed."}),
                "mode": (PRUNE_MODES, {
                    "default": "disabled",
                    "tooltip": "disabled: do nothing. after_assembly: delete only once the "
                               "final video exists and is non-empty. always: delete regardless.",
                }),
                "expected_frames": ("INT", {"default": 0, "min": 0, "max": 10 ** 7,
                                            "tooltip": "0 = no check. Otherwise refuse to delete "
                                                       "unless the count matches exactly. Feed "
                                                       "frame_count from Frame Store Status."}),
                "prune_video_intermediates": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also remove the intermediates VideoCombine leaves next to the "
                               "muxed result (the silent .mp4, the metadata png).",
                }),
            },
            "optional": {
                "filenames": ("VHS_FILENAMES", {
                    "tooltip": "Filenames output of the final Video Combine. Required in "
                               "after_assembly mode: it is what forces the assembly to run "
                               "before this node, and what proves the video exists.",
                }),
            },
        }

    CATEGORY = "Helpers 🧰/io"

    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("deleted", "final_video", "report")

    FUNCTION = "prune"
    OUTPUT_NODE = True

    DESCRIPTION = ("Delete a chunk pipeline's working frames once the final video exists. "
                   "Does nothing until mode says otherwise.")

    def prune(self, directory, prefix, mode, expected_frames,
              prune_video_intermediates, filenames=None):
        path = resolve_store_directory(directory)  # raises on an escaping path
        final_video, intermediates = split_filenames(filenames)
        report = []
        deleted = 0

        if mode == "disabled":
            return (0, final_video or "", "mode is disabled: nothing deleted")

        if mode == "after_assembly":
            if final_video is None:
                return (0, "", "after_assembly: no filenames connected, so there is no proof "
                               "the final video was written. Nothing deleted.")
            if not os.path.isfile(final_video) or os.path.getsize(final_video) == 0:
                return (0, final_video, f"after_assembly: {final_video} is missing or empty, "
                                        "so the assembly did not succeed. Nothing deleted.")
            report.append(f"final video kept: {final_video}")

        present = list_frame_files(path, prefix)
        if expected_frames > 0 and len(present) != expected_frames:
            return (0, final_video or "",
                    f"expected {expected_frames} frames but found {len(present)} in {path}. "
                    "Nothing deleted -- the counts must match exactly.")

        for _, name in present:
            try:
                os.remove(os.path.join(path, name))
                deleted += 1
            except OSError as e:
                report.append(f"could not delete {name}: {e}")
        report.append(f"{deleted} frame(s) removed from {path}")

        # only ever remove the folder itself if it is empty: never rmtree
        try:
            if not os.listdir(path):
                os.rmdir(path)
                report.append("empty folder removed")
            else:
                report.append("folder kept: it still holds files we don't own")
        except OSError:
            pass

        if prune_video_intermediates and intermediates:
            for candidate in intermediates:
                if not is_under_comfy_dirs(candidate):
                    report.append(f"skipped {candidate}: outside output/ and temp/")
                    continue
                try:
                    os.remove(candidate)
                    deleted += 1
                    report.append(f"intermediate removed: {os.path.basename(candidate)}")
                except OSError as e:
                    report.append(f"could not delete {candidate}: {e}")

        return (deleted, final_video or "", "\n".join(report))


# --------------------------------------------------------------------------
# D. save frame sequence
# --------------------------------------------------------------------------

def next_free_index(path, prefix):
    entries = list_frame_files(path, prefix)
    return entries[-1][0] + 1 if entries else 0


def save_frame_sequence(images, path, prefix, format="png", start_index=-1,
                        metadata=None, digits=6):
    """Write a batch as numbered stills, one atomic file at a time."""
    spec = SEQUENCE_FORMATS[format]
    os.makedirs(path, exist_ok=True)
    index = next_free_index(path, prefix) if start_index < 0 else int(start_index)

    pbar = ProgressBar(len(images))
    written = 0
    for image in images:
        array = np.clip(image[..., :3].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(array)

        options = dict(spec["options"])
        if metadata and spec["pil_format"] == "PNG":
            info = PngImagePlugin.PngInfo()
            for key, value in metadata.items():
                info.add_text(key, json.dumps(value))
            options["pnginfo"] = info

        name = f"{prefix}_{index:0{digits}d}{spec['extension']}"
        final_path = os.path.join(path, name)
        # write beside the target then rename: an interrupted run leaves a .tmp,
        # never a half-written frame the status node would have to sort out
        temp_path = final_path + ".tmp"
        pil_image.save(temp_path, format=spec["pil_format"], **options)
        os.replace(temp_path, final_path)

        index += 1
        written += 1
        pbar.update(1)

    return written, index


class SaveFrameSequence:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "directory": ("STRING", {"default": "H3_inpaint",
                                         "tooltip": "Folder under ComfyUI's output/."}),
                "prefix": ("STRING", {"default": "frame"}),
                "format": (list(SEQUENCE_FORMATS), {"default": "png"}),
                "embed_workflow": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save Image embeds the whole workflow in every file, which is "
                               "what makes a long sequence heavy. Off by default here; PNG only.",
                }),
            },
            "optional": {
                "start_index": ("INT", {"default": -1, "min": -1, "max": 10 ** 7,
                                        "tooltip": "-1 continues after the highest index already "
                                                   "in the folder, which is what a resumable run "
                                                   "wants. Otherwise writes from this index."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "Helpers 🧰/io"

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("written", "next_index", "directory")

    FUNCTION = "save"
    OUTPUT_NODE = True

    DESCRIPTION = ("Write an IMAGE batch as a numbered still sequence, appending to what is "
                   "already there. Atomic writes, no workflow embedded by default.")

    def save(self, images, directory, prefix, format, embed_workflow,
             start_index=-1, prompt=None, extra_pnginfo=None):
        path = resolve_store_directory(directory)

        metadata = None
        if embed_workflow:
            metadata = {}
            if prompt is not None:
                metadata["prompt"] = prompt
            for key, value in (extra_pnginfo or {}).items():
                metadata[key] = value

        written, next_index = save_frame_sequence(images, path, prefix, format,
                                                  start_index, metadata)
        return (written, next_index, path)


NODE_CLASS_MAPPINGS = {
    "CMDR_H3ChunkPlanner": H3ChunkPlanner,
    "CMDR_FrameStoreStatus": FrameStoreStatus,
    "CMDR_FrameStorePrune": FrameStorePrune,
    "CMDR_SaveFrameSequence": SaveFrameSequence,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CMDR_H3ChunkPlanner": "H3 Chunk Planner 🧰",
    "CMDR_FrameStoreStatus": "Frame Store Status 🧰",
    "CMDR_FrameStorePrune": "Frame Store Prune 🧰",
    "CMDR_SaveFrameSequence": "Save Frame Sequence 🧰",
}
