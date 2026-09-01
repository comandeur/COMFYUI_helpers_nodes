from .load_animation_nodes import (NODE_CLASS_MAPPINGS as _load_animation_classes,
                                   NODE_DISPLAY_NAME_MAPPINGS as _load_animation_names)
from .rtx_vsr_nodes import (NODE_CLASS_MAPPINGS as _rtx_vsr_classes,
                            NODE_DISPLAY_NAME_MAPPINGS as _rtx_vsr_names)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _classes, _names in ((_load_animation_classes, _load_animation_names),
                         (_rtx_vsr_classes, _rtx_vsr_names)):
    NODE_CLASS_MAPPINGS.update(_classes)
    NODE_DISPLAY_NAME_MAPPINGS.update(_names)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
