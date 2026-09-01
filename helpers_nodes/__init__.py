from .load_animation_nodes import (NODE_CLASS_MAPPINGS as _load_animation_classes,
                                   NODE_DISPLAY_NAME_MAPPINGS as _load_animation_names)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(_load_animation_classes)
NODE_DISPLAY_NAME_MAPPINGS.update(_load_animation_names)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
