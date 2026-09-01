"""Load-format presets.

Mirrors VideoHelperSuite's ``VHSLoadFormats`` so the ``format`` widget offers the
same choices, and so third party packs that registered extra formats on
``nodes.VHSLoadFormats`` show up here too.

    target_rate: fps the format expects (used by the frontend reset button)
    dim:         (downscale_ratio, mod, default_width, default_height)
    frames:      (divisor, remainder[, strict]) constraint on the frame count
"""
import nodes

HELPER_LOAD_FORMATS = {
    'None': {},
    'AnimateDiff': {'target_rate': 8, 'dim': (8, 0, 512, 512)},
    'Mochi': {'target_rate': 24, 'dim': (16, 0, 848, 480), 'frames': (6, 1)},
    'LTXV': {'target_rate': 24, 'dim': (32, 0, 768, 512), 'frames': (8, 1)},
    'Hunyuan': {'target_rate': 24, 'dim': (16, 0, 848, 480), 'frames': (4, 1)},
    'Cosmos': {'target_rate': 24, 'dim': (16, 0, 1280, 704), 'frames': (8, 1)},
    'Wan': {'target_rate': 16, 'dim': (8, 0, 832, 480), 'frames': (4, 1)},
}

if not hasattr(nodes, 'VHSLoadFormats'):
    nodes.VHSLoadFormats = {}


def get_load_formats():
    formats = {}
    formats.update(nodes.VHSLoadFormats)
    formats.update(HELPER_LOAD_FORMATS)
    return (list(formats.keys()),
            {'default': 'AnimateDiff', 'formats': formats})


def get_format(format):
    if format in HELPER_LOAD_FORMATS:
        return HELPER_LOAD_FORMATS[format]
    return nodes.VHSLoadFormats.get(format, {})
