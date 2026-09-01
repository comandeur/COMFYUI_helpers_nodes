# COMFYUI_helpers_nodes

A small pack of ComfyUI nodes I keep needing. Nodes live under the **Helpers 🧰**
category.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/comandeur/COMFYUI_helpers_nodes
```

Restart ComfyUI. No extra dependencies beyond what ComfyUI already ships
(Pillow, numpy, torch, psutil).

### Node ids

Node ids are prefixed `CMDR_` rather than something generic, because
`NODE_CLASS_MAPPINGS` is a single flat namespace shared by every installed pack:
two packs claiming the same key overwrite each other silently, and workflows
saved with either one then resolve to whichever loaded last.

The pre-0.4.0 `Helpers_` ids are still registered as `DEPRECATED` aliases, so
workflows saved before the rename keep working. They are hidden from the node
search; open such a workflow, drop in the `CMDR_` node, and the alias can go.

---

## Load GIF/WebP (Upload) 🧰

`CMDR_LoadAnimationUpload`

A GIF / animated WebP / APNG loader shaped exactly like VideoHelperSuite's
*Load Video (Upload)*: same widgets, same four outputs, same socket types — so
it drops straight into an existing video workflow. Click **choose animation to
upload** or drop a file onto the node; the picked animation plays in the node
preview.

### Inputs

| widget | meaning |
| --- | --- |
| `image` | file in `ComfyUI/input` (`.gif`, `.webp`, `.apng`) |
| `force_rate` | resample to this fps. `0` keeps the animation's own timing |
| `custom_width` / `custom_height` | `0` keeps the source size; set one to scale by aspect |
| `frame_load_cap` | stop after N loaded frames (`0` = no cap) |
| `skip_first_frames` | drop N frames from the start (after rate conversion) |
| `select_every_nth` | keep 1 frame out of N |
| `meta_batch` *(optional)* | VideoHelperSuite `Meta Batch Manager`, for streaming long animations |
| `vae` *(optional)* | encode frames straight to LATENT instead of returning IMAGE |
| `format` *(optional)* | same presets as VHS (AnimateDiff, Wan, LTXV…): sets the size rounding and the frame-count constraint |
| `alpha` *(optional)* | `composite_black` (default), `composite_white`, or `keep_alpha` for a 4-channel IMAGE |
| `audio_output` *(optional)* | see below |

### Outputs

`IMAGE` (or `LATENT` when a VAE is connected) · `frame_count` · `audio` ·
`video_info`

`video_info` is a `VHS_VIDEOINFO` dict with the same keys VideoHelperSuite uses
(`source_fps`, `source_frame_count`, `source_duration`, `source_width`,
`source_height`, `loaded_*`), so it plugs into their *Video Info* node.

### The audio output

GIF and WebP carry no audio track, so the `audio` output is a stand-in:

* `none` (default) — outputs a real `None`. Leave the socket unconnected, or
  feed it to a node that treats audio as optional (e.g. VHS *Video Combine*,
  which then writes a video with no audio stream).
* `silent` — outputs a valid, silent stereo `AUDIO` at 44.1 kHz, exactly as long
  as the loaded frames. Use this when a downstream node insists on receiving
  real audio.

### Notes on timing

GIF and WebP store a delay *per frame*, so there is no single source fps. The
node reads the whole delay timeline, reports the average as `source_fps`, and
`force_rate` resamples against real timestamps rather than assuming a constant
rate. Delays under 20 ms are treated as 100 ms, which is what browsers do with
the "as fast as possible" delays most GIF encoders write.

Alpha is composited by Pillow with the GIF disposal method / WebP blending
applied, so transparent frames come out looking like they do in a viewer instead
of leaving trails.

---

## RTX Video Upscale (IMAGE) 🧰

`CMDR_RTXVideoUpscale`

NVIDIA RTX Video Super Resolution applied to an `IMAGE` batch, in place in the
workflow: **IMAGE + AUDIO in, upscaled IMAGE + the same AUDIO out.** No video
file, no ffmpeg round trip, nothing written to disk.

[ComfyUI-RTX-Video-Suite](https://github.com/uczensokratesa/ComfyUI-RTX-Video-Suite)
wraps the same SDK, but around files: it reads a video off disk and writes an
upscaled one back, which means leaving the graph and re-encoding every time.
This node keeps the model parameters on the node and everything else in the
workflow, so it can sit between any two video nodes.

### Requirements

* an NVIDIA RTX GPU
* the NVIDIA Video Effects (VFX) SDK — an importable `nvvfx` in the python
  environment ComfyUI runs on

Without the SDK the node still loads; it raises a readable error when executed.

### Inputs

| widget | meaning |
| --- | --- |
| `images` | the frames to upscale |
| `quality` | the full SDK quality list, read from the installed `nvvfx` |
| `scale` | output multiplier, used when both custom dimensions are `0` |
| `custom_width` / `custom_height` | explicit output size; `0` = derive from `scale`, set one to scale by aspect |
| `audio` *(optional)* | passed straight through, untouched |
| `align` *(optional)* | `even` (default), `exact`, or `multiple of 8` |
| `keep_model_loaded` *(optional)* | keep the inference engine in VRAM between runs (default on) |

### Quality levels

The list comes from the SDK itself, so it covers more than upscaling:

* `BICUBIC`, `LOW`…`ULTRA` — standard upscaling, also removes compression
  artifacts
* `HIGHBITRATE_LOW`…`HIGHBITRATE_ULTRA` — upscaling for clean/lossless sources,
  skips artifact suppression so it doesn't soften detail that's already good
* `DENOISE_*` and `DEBLUR_*` — enhancement at source resolution. `scale` and the
  custom dimensions are ignored for these, since the SDK requires output size to
  equal input size.

### Notes

Frames are handed to the SDK one at a time on the GPU over DLPack, so VRAM cost
is one input frame plus one output frame no matter how long the batch is — the
batch itself still lives in RAM, as any `IMAGE` does.

The inference engine is loaded once and cached across executions (a first run
costs about a second, later runs are instant). Quality and output size can be
changed without reloading it. Turn `keep_model_loaded` off to release it after
each run.

`align` exists because the SDK accepts any output size, including odd numbers,
but h264/h265 need even dimensions — so the default rounds to even. Use
`multiple of 8` if the result goes back through a VAE, or `exact` to keep the
aspect ratio exactly.

### Memory

An `IMAGE` output has to exist as one contiguous tensor — that is the socket's
contract, not a choice this node makes. 1000 frames of 1280x704 in float32 is
10.5 GiB of RAM, held alongside the input batch, and any IMAGE-based path pays
it: *Load Video → anything → Video Combine* costs the same. When it doesn't fit:

* run the graph through a VHS **Meta Batch Manager**, so only `frames_per_batch`
  frames exist at a time. This node is stateless across batches and keeps its
  engine loaded between them, so it costs nothing to use that way;
* set `output_dtype` to `float16`, which halves the figure;
* or use **RTX Video Upscale (to file)** below, which never allocates it at all.

---

## RTX Video Upscale (to file) 🧰

`CMDR_RTXVideoUpscaleToFile`

The same upscale, encoded straight to a video file instead of an `IMAGE` batch.
Each frame is handed to the encoder and dropped, so the run costs one frame of
RAM no matter how long the video is. The result plays in a preview on the node.

Encoding goes through PyAV, which already ships with ComfyUI — no ffmpeg binary
on PATH, no subprocess pipe.

| widget | meaning |
| --- | --- |
| `images` | the frames to upscale |
| `quality` / `scale` / `custom_width` / `custom_height` / `align` | identical to the IMAGE node |
| `frame_rate` | fps written to the file — feed it from the loader's `video_info` to keep the original timing |
| `filename_prefix` | path under `output/`, supports ComfyUI's `%date:yyyy-MM-dd%` tokens |
| `format` | `mp4` or `avi`, both h264 so `crf` keeps its meaning |
| `crf` | 0 lossless, 18 visually lossless, 23 default, 51 worst |
| `save_output` | on: written to `output/`. off: written to `temp/`, so it still plays in the preview but isn't kept |
| `save_metadata` | embed the prompt and workflow in the file |
| `audio` *(optional)* | muxed in (AAC in mp4, MP3 in avi); leave unconnected for a silent file |

Output: the full path as a `STRING`.

Metadata is written the same way ComfyUI's own save nodes write it — the prompt
and the workflow as JSON tags, so dropping the file back into ComfyUI restores
the graph. That works reliably in mp4 (the muxer is told to keep custom tags);
avi's tag support is poor, so treat it as best-effort there.

---

## Scale Resolution to Megapixels 🧰

`CMDR_ScaleResolutionToMegapixels`

Takes a `width`/`height` pair — typically straight out of a resolution picker —
keeps its aspect ratio, and resizes it to a megapixel budget. Two INT outputs,
`width` and `height`, ready to feed an empty latent or a resize node.

| widget | meaning |
| --- | --- |
| `width` / `height` | the resolution whose ratio you want to keep; connect them or type them |
| `megapixels` | pixel budget for the result (`1.0` = one million pixels) |
| `multiple` | snap both dimensions to a multiple of this (default `32`) |

The ratio is read from the two numbers exactly as they arrive, so if they were
already snapped by the picker, that snapped ratio is what gets preserved.

Rounding each side to its own nearest multiple lets the ratio drift — 1920x1080
at 0.5 MP on a multiple of 32 lands on 1.71 instead of 1.78. This node scores all
four floor/ceil combinations instead and picks the closest ratio, using the pixel
count only as a tie-break. So the megapixel value is a target, not a hard
ceiling: expect the result to land within a few percent of it, and closer as
`multiple` gets smaller.

`megapixel_base` (optional) says how many pixels one megapixel is worth:
`1,000,000` by default, or `1024x1024` (1 048 576) to match ComfyUI's built-in
Resolution Selector, which counts that way — a 4.9% bigger area for the same
number.

---

## Resolution Selector 🧰

`CMDR_ResolutionSelector`

The same maths driven by a preset instead of a width/height pair: pick an aspect
ratio and a megapixel budget, get `width` and `height`.

A more precise replacement for ComfyUI's built-in *Resolution Selector*:

* `megapixels` moves in steps of **0.01** instead of 0.1, and starts at 0.01
  instead of 0.1;
* two presets the built-in doesn't have — **9:21 (Portrait Ultrawide)** and the
  **5:6 / 6:5** near-square pair — for eleven in total, core's eight included and
  spelled identically;
* the ratio survives the rounding. Core rounds each side to its own nearest
  multiple, which drifts: 16:9 at 0.9 MP on a multiple of 32 gives 1280x736, a
  ratio of 1.739 — 2.2% off. This node gives 1312x736, i.e. 1.783, 0.27% off;
* presets are ordered square-first then outward, so the pairs read in order.

Set `megapixel_base` to `1024x1024` for a drop-in swap: the target area is then
core's, and only the rounding differs — never in the ratio's disfavour, which the
tests check across every preset, budget and multiple.

---

# Chunked long-video pipeline

Four nodes for processing a long source a clip at a time, across several runs,
with the frames living in an output folder in between. Written for MiniMax H3
inpainting; nothing is H3-specific except the length grid. They live under
**Helpers 🧰/h3** and **Helpers 🧰/io**.

The loop they close:

```
Frame Store Status ──chunks_done──► H3 Chunk Planner ──length, skip_frames──► Load Video
        ▲                                   │
        │                           trim_start, trim_length ──► ImageFromBatch ──► …
        │                                   │                            │
        └────── total_frames ───────────────┤                            ▼
                                            └── total_frames ──► Save Frame Sequence
                                                 (as stop_at_frames)

[assembly, once every chunk is done]
Load Images (Path) ──► Video Combine ──Filenames──► Frame Store Prune
```

Status reads the folder, so `chunk_index` stops being something you manage by
hand: queue the workflow M times and each run picks up where the last stopped.

**Wire `total_frames` from the planner into both Status and Save.** Without it
Status cannot tell a finished job from a partial chunk — see below — and an
over-long queue appends duplicate frames. If the loop back into Status bothers
you, feed its `total_frames` from the same `source_frames` / `video_info` the
planner uses instead; it is the same number.

## H3 Chunk Planner 🧰

`CMDR_H3ChunkPlanner`

Turns a wanted clip duration into the exact windows to load. Replaces a chain of
nine math nodes.

| widget | meaning |
| --- | --- |
| `clip_seconds` / `fps` | wanted duration; the real length is rounded **up** onto H3's `17k+5` grid |
| `chunk_index` | which chunk to plan — feed it `chunks_done` from Frame Store Status |
| `source_frames` | total frames of the source, ignored when `video_info` is connected |
| `video_info` *(optional)* | a `VHS_VIDEOINFO`: the count becomes `source_duration × fps` |
| `clamp_to_trained_range` *(optional)* | keep the length inside H3's trained `[124, 362]` |

Outputs: `length` → `frame_load_cap`, `skip_frames` → `skip_first_frames`,
`trim_start` / `trim_length` → `ImageFromBatch`, plus `chunk_count`, `is_last`,
a readable `info`, and `total_frames` — the count it actually used, which is the
only way to read the number when it came from `video_info`.

The last chunk's window is pulled back to end exactly on the source's final
frame, and the overlap that creates is trimmed off the front of what's kept. The
guarantee, checked over 16752 combinations in the tests: concatenating the kept
ranges for `k` in `0..M-1` reproduces `range(T)` exactly — no gap, no duplicate
frame. Asking for a `clip_seconds` longer than the source is an error rather
than a silently off-grid frame count; asking for a chunk past the last one is
clamped to the last, and `info` says so.

## Frame Store Status 🧰

`CMDR_FrameStoreStatus`

Reads a frame folder and says where to resume: `frame_count`, `chunks_done`
(the resume index), `is_clean`, a `report`, and `is_complete`.

**Wire `total_frames`.** The last chunk is shorter than the others — it writes
`length - overlap` frames — so a finished job is never a whole multiple of
`length`. Judging by `frame_count // length` alone, a finished 1440-frame job in
chunks of 158 reads as 9 chunks instead of 10: the loop would redo the last chunk
on every further run, appending duplicates, and `auto_repair` would delete those
18 perfectly good frames as if they were a partial chunk. With `total_frames`
connected, reaching it means done: `chunks_done` becomes the full count,
`is_complete` goes true, and `auto_repair` will not touch the store.

At `total_frames = 0` the node keeps the older, count-only behaviour, so nothing
that was wired before changes — but that is the mode with the trap in it.

`verify_integrity` checks each file's end marker (`IEND` for PNG, the RIFF size
for WebP, `FFD9` for JPEG), so a run interrupted mid-write isn't counted as done.
`auto_repair` then deletes the incomplete files, and the tail of a partial chunk
to land back on a whole multiple of `length` — but never on a store that already
reached `total_frames`, and never when there is a hole in the numbering, since
trimming the tail cannot fill a hole. Frames *past* `total_frames` are reported
and left alone: deleting them would be guessing which copy is the good one.

`IS_CHANGED` returns `nan` on purpose. Without it ComfyUI would cache the node
and every run in a queue of ten would read the same `chunks_done` — the resume
would stop advancing. A missing folder is a starting state, not an error.

## Frame Store Prune 🧰

`CMDR_FrameStorePrune`

Deletes the working frames once the final video exists. `mode` defaults to
`disabled`, so a shared workflow never deletes anything by surprise.

In `after_assembly` mode the `filenames` input (the final Video Combine's
`Filenames`) is mandatory: it is what makes the assembly run *before* this node —
ComfyUI has no notion of "after", only data dependencies — and what proves the
video was written. Nothing is deleted unless that file exists and is non-empty,
and unless `expected_frames` (feed it `frame_count`) matches exactly.

`prune_video_intermediates` also removes what Video Combine leaves beside the
muxed result: `VHS_FILENAMES` is `(save_output, [paths…])` whose **last** entry
is the real file and whose earlier entries are intermediates — the silent `.mp4`
of an audio mux, the metadata png. Every path is checked to be inside `output/`
or `temp/` before removal, files are deleted one by one filtered on `prefix`
(never `rmtree`), and the folder itself goes only if it ends up empty.

## Save Frame Sequence 🧰

`CMDR_SaveFrameSequence`

Writes an `IMAGE` batch as numbered stills, appending to whatever is already
there. Replaces `SaveImage` for inter-run storage: `SaveImage` embeds the whole
workflow JSON in *every* file, which over a few thousand frames is most of what
you're writing, and only speaks PNG.

`format` covers `png`, `webp_lossless`, `webp_q95`, `jpeg_q95`;
`embed_workflow` is off by default (and PNG-only). `start_index` at `-1`
continues after the highest index present, which is what a resumable run wants.

`stop_at_frames` is the safety catch: once the folder holds that many frames the
node writes nothing and returns `written = 0`. Feed it the planner's
`total_frames` and a queue left running longer than the job wastes GPU time
instead of corrupting the sequence. It counts the folder anyway to resolve
`start_index = -1`, so this costs nothing.

Files are numbered on six digits — five would cap at 99 999 frames, about 69
minutes at 24fps — and written to a `.tmp` then `os.replace`d, so an interrupted
run leaves a stray temp file rather than a truncated frame.

## Tests

```bash
python_embeded/python.exe custom_nodes/COMFYUI_helpers_nodes/tests/test_chunk_nodes.py
```

Runs without starting ComfyUI: `folder_paths` is stubbed onto a throwaway
directory, so the file-touching nodes are exercised for real without going near
your `output/`.
