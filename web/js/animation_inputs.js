// Older saved workflows can contain null instead of numeric widget values.
// Restore only missing/non-finite values; connected inputs remain untouched.
export function restoreAnimationNumbers(node, inputs) {
    for (const [name, [type, options]] of Object.entries(inputs)) {
        if (type !== "INT" && type !== "FLOAT") continue;
        if (node.inputs?.some((input) => input.name === name && input.link != null)) continue;
        const widget = node.widgets?.find((widget) => widget.name === name);
        if (!widget) continue;
        if (widget.value == null ||
            (typeof widget.value === "number" && !Number.isFinite(widget.value))) {
            widget.value = options.default;
        }
    }
}
