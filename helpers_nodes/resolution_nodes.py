"""Resolution helpers.

Pure integer maths, no torch or ComfyUI imports: these nodes only shuffle
numbers around before anything gets allocated.
"""
import math

MAX_DIM = 16384


def snap_candidates(value, multiple):
    """The multiples of ``multiple`` just below and just above ``value``."""
    low = max(multiple, int(math.floor(value / multiple)) * multiple)
    high = max(multiple, int(math.ceil(value / multiple)) * multiple)
    return sorted({low, high})


def scale_to_megapixels(width, height, megapixels, multiple=32):
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
    target_pixels = max(1.0, megapixels * 1_000_000.0)

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
                                         "step": 0.05,
                                         "tooltip": "Pixel budget for the result. "
                                                    "1.0 = 1 million pixels."}),
                "multiple": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1,
                                     "tooltip": "Snap both dimensions to a multiple of "
                                                "this. Larger values trade a little "
                                                "ratio accuracy for model-friendly sizes."}),
            },
        }

    CATEGORY = "Helpers 🧰"

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    FUNCTION = "scale"

    DESCRIPTION = ("Keep the aspect ratio of a width/height pair, resize it to a "
                   "megapixel budget, snapped to a multiple.")

    def scale(self, width, height, megapixels, multiple):
        return scale_to_megapixels(width, height, megapixels, multiple)


NODE_CLASS_MAPPINGS = {
    "Helpers_ScaleResolutionToMegapixels": ScaleResolutionToMegapixels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Helpers_ScaleResolutionToMegapixels": "Scale Resolution to Megapixels 🧰",
}
