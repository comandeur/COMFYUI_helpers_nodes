"""Resolution helpers.

Pure integer maths, no torch or ComfyUI imports: these nodes only shuffle
numbers around before anything gets allocated.
"""
import math

MAX_DIM = 16384

# Square first, then each portrait/landscape pair, going outward from square.
# Core's own picker orders 2:3 before 3:4, which reads oddly; this doesn't.
# Ratios are exact integers, so 3:2 really is 1.5, not 1.4985.
ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "5:6 (Portrait Near-Square)": (5, 6),
    "6:5 (Near-Square)": (6, 5),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "9:21 (Portrait Ultrawide)": (9, 21),
    "21:9 (Ultrawide)": (21, 9),
}

# What one "megapixel" counts as. A megapixel is a million pixels, but core's
# own Resolution Selector uses 1024x1024 (1048576) -- a 4.9% larger area for the
# same number -- so the choice is exposed rather than guessed.
MEGAPIXEL_BASES = {
    "1,000,000": 1_000_000,
    "1024x1024": 1024 * 1024,
}
MEGAPIXEL_BASE_TOOLTIP = ("Pixels per megapixel. 1,000,000 is what the word means; "
                          "1024x1024 is what ComfyUI's built-in Resolution Selector "
                          "uses, so pick it to reproduce that node's numbers.")


def snap_candidates(value, multiple):
    """The multiples of ``multiple`` just below and just above ``value``."""
    low = max(multiple, int(math.floor(value / multiple)) * multiple)
    high = max(multiple, int(math.ceil(value / multiple)) * multiple)
    return sorted({low, high})


def scale_to_megapixels(width, height, megapixels, multiple=32,
                        megapixel_base="1,000,000"):
    """Same aspect ratio as ``width``/``height``, resized to hit a pixel budget.

    The ratio is taken from the two numbers as given -- if they came out of a
    resolution picker already snapped to a multiple, that snapped ratio is what
    gets preserved, which is what the model actually saw.

    Rounding each side to its own nearest multiple would let the ratio drift by
    several percent (1920x1080 at 0.5MP on a multiple of 32 lands on 1.71
    instead of 1.78), so all four floor/ceil combinations are scored instead:
    closest ratio wins, closest pixel count breaks the tie. That makes the
    megapixel value a target rather than a hard ceiling.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    multiple = max(1, int(multiple))

    ratio = width / height
    target_pixels = max(1.0, megapixels * MEGAPIXEL_BASES[megapixel_base])

    exact_height = math.sqrt(target_pixels / ratio)
    exact_width = exact_height * ratio

    best_score = None
    best_size = None
    for new_width in snap_candidates(exact_width, multiple):
        for new_height in snap_candidates(exact_height, multiple):
            ratio_error = abs(new_width / new_height - ratio) / ratio
            pixel_error = abs(new_width * new_height - target_pixels) / target_pixels
            score = (round(ratio_error, 9), pixel_error)
            if best_score is None or score < best_score:
                best_score = score
                best_size = (new_width, new_height)
    return best_size


class ScaleResolutionToMegapixels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 1, "max": MAX_DIM, "step": 1,
                                  "tooltip": "Connect the width of the resolution you "
                                             "want to keep the ratio of."}),
                "height": ("INT", {"default": 1216, "min": 1, "max": MAX_DIM, "step": 1}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0,
                                         "step": 0.01,
                                         "tooltip": "Pixel budget for the result. "
                                                    "1.0 = 1 million pixels."}),
                "multiple": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1,
                                     "tooltip": "Snap both dimensions to a multiple of "
                                                "this. Larger values trade a little "
                                                "ratio accuracy for model-friendly sizes."}),
            },
            "optional": {
                "megapixel_base": (list(MEGAPIXEL_BASES), {"default": "1,000,000",
                                                           "tooltip": MEGAPIXEL_BASE_TOOLTIP}),
            },
        }

    CATEGORY = "Helpers 🧰"

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    FUNCTION = "scale"

    DESCRIPTION = ("Keep the aspect ratio of a width/height pair, resize it to a "
                   "megapixel budget, snapped to a multiple.")

    def scale(self, width, height, megapixels, multiple, megapixel_base="1,000,000"):
        return scale_to_megapixels(width, height, megapixels, multiple, megapixel_base)


class ResolutionSelector:
    """Same idea, but the ratio comes from a preset instead of a width/height pair."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "aspect_ratio": (list(ASPECT_RATIOS), {"default": "1:1 (Square)"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0,
                                         "step": 0.01,
                                         "tooltip": "Pixel budget for the result. "
                                                    "1.0 = 1 million pixels."}),
                "multiple": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1,
                                     "tooltip": "Snap both dimensions to a multiple of "
                                                "this. Larger values trade a little "
                                                "ratio accuracy for model-friendly sizes."}),
            },
            "optional": {
                "megapixel_base": (list(MEGAPIXEL_BASES), {"default": "1,000,000",
                                                           "tooltip": MEGAPIXEL_BASE_TOOLTIP}),
            },
        }

    CATEGORY = "Helpers 🧰"

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    FUNCTION = "select"

    DESCRIPTION = ("Pick an aspect ratio and a megapixel budget, get a width and a height "
                   "snapped to a multiple.")

    def select(self, aspect_ratio, megapixels, multiple, megapixel_base="1,000,000"):
        ratio_width, ratio_height = ASPECT_RATIOS[aspect_ratio]
        return scale_to_megapixels(ratio_width, ratio_height, megapixels, multiple,
                                   megapixel_base)


class LegacyScaleResolutionToMegapixels(ScaleResolutionToMegapixels):
    """The pre-CMDR_ node id, so workflows saved with it still resolve."""
    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "CMDR_ScaleResolutionToMegapixels": ScaleResolutionToMegapixels,
    "CMDR_ResolutionSelector": ResolutionSelector,
    "Helpers_ScaleResolutionToMegapixels": LegacyScaleResolutionToMegapixels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CMDR_ScaleResolutionToMegapixels": "Scale Resolution to Megapixels 🧰",
    "CMDR_ResolutionSelector": "Resolution Selector 🧰",
    "Helpers_ScaleResolutionToMegapixels": "Scale Resolution to Megapixels 🧰 (old id)",
}
