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

---

## Load GIF/WebP (Upload) 🧰

`Helpers_LoadAnimationUpload`

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
