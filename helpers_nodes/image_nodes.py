"""Image resizing helpers."""
import comfy.utils

from .resolution_nodes import (MEGAPIXEL_BASES, MEGAPIXEL_BASE_TOOLTIP,
                               cap_to_megapixels)

UPSCALE_METHODS = ["lanczos", "area", "bicubic", "bilinear", "nearest-exact"]


class LimitImageMegapixels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0,
                                             "step": 0.01,
                                             "tooltip": "Ceiling for the image area. "
                                                        "Images at or below it pass "
                                                        "through untouched."}),
                "resize_method": (UPSCALE_METHODS, {"default": "lanczos",
                    "tooltip": "Filter used when shrinking. lanczos and area both "
                               "downscale cleanly; area is faster on big batches."}),
            },
            "optional": {
                "megapixel_base": (list(MEGAPIXEL_BASES), {"default": "1,000,000",
                                                           "tooltip": MEGAPIXEL_BASE_TOOLTIP}),
            },
        }

    CATEGORY = "Helpers 🧰"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "limit"

    DESCRIPTION = ("Shrink an image, keeping its aspect ratio, so it is no larger than "
                   "max_megapixels. Images already within the limit pass through "
                   "unchanged; nothing is ever upscaled.")

    def limit(self, image, max_megapixels, resize_method,
              megapixel_base="1,000,000"):
        height, width = image.shape[1], image.shape[2]
        new_width, new_height, resized = cap_to_megapixels(
            width, height, max_megapixels, 1, megapixel_base)
        if resized:
            samples = image.movedim(-1, 1)
            samples = comfy.utils.common_upscale(samples, new_width, new_height,
                                                 resize_method, "disabled")
            image = samples.movedim(1, -1)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "CMDR_LimitImageMegapixels": LimitImageMegapixels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CMDR_LimitImageMegapixels": "Limit Image Megapixels 🧰",
}
