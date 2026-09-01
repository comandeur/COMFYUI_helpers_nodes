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
