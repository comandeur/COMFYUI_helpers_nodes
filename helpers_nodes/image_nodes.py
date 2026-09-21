"""Image resizing helpers."""
import comfy.utils

from .resolution_nodes import (MEGAPIXEL_BASES, MEGAPIXEL_BASE_TOOLTIP,
                               cap_to_megapixels)

UPSCALE_METHODS = ["lanczos", "area", "bicubic", "bilinear", "nearest-exact"]

# Slots the node can grow to. The frontend (web/js/image_slots.js) only shows
# the connected ones plus one empty slot; Python has to declare them all.
MAX_IMAGES = 16
IMAGE_SLOTS = ["image"] + [f"image_{i}" for i in range(2, MAX_IMAGES + 1)]


class LimitImageMegapixels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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
                **{name: ("IMAGE", {"tooltip": "Optional: an empty slot outputs "
                                                "nothing (None)."})
                   for name in IMAGE_SLOTS},
                "megapixel_base": (list(MEGAPIXEL_BASES), {"default": "1,000,000",
                                                           "tooltip": MEGAPIXEL_BASE_TOOLTIP}),
            },
        }

    CATEGORY = "Helpers 🧰"

    RETURN_TYPES = ("IMAGE",) * MAX_IMAGES
    RETURN_NAMES = tuple(IMAGE_SLOTS)

    FUNCTION = "limit"

    DESCRIPTION = ("Shrink each image, keeping its aspect ratio, so it is no larger than "
                   "max_megapixels. Images already within the limit pass through "
                   "unchanged; nothing is ever upscaled. A new image slot appears as "
                   "soon as the last one is connected; every slot is optional and an "
                   "empty one outputs None.")

    def limit(self, max_megapixels, resize_method, megapixel_base="1,000,000",
              **images):
        return tuple(self.limit_one(images.get(name), max_megapixels, resize_method,
                                    megapixel_base)
                     for name in IMAGE_SLOTS)

    @staticmethod
    def limit_one(image, max_megapixels, resize_method, megapixel_base):
        if image is None:
            return None
        height, width = image.shape[1], image.shape[2]
        new_width, new_height, resized = cap_to_megapixels(
            width, height, max_megapixels, 1, megapixel_base)
        if not resized:
            return image
        samples = image.movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, new_width, new_height,
                                             resize_method, "disabled")
        return samples.movedim(1, -1)


NODE_CLASS_MAPPINGS = {
    "CMDR_LimitImageMegapixels": LimitImageMegapixels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CMDR_LimitImageMegapixels": "Limit Image Megapixels 🧰",
}
