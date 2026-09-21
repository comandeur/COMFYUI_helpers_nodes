// Grow/shrink a node's IMAGE slots so there is always exactly one empty input
// after the last connected one, with a matching output for every input.
// Slot names: "image", "image_2", "image_3", ... (must match Python's IMAGE_SLOTS).

export function slotName(index) {
    return index === 0 ? "image" : `image_${index + 1}`;
}

function slotIndex(name) {
    if (name === "image") return 0;
    const m = /^image_(\d+)$/.exec(name ?? "");
    return m ? Number(m[1]) - 1 : -1;
}

export function syncImageSlots(node, max) {
    const inputs = node.inputs ?? [];
    const outputs = node.outputs ?? [];

    let lastConnected = -1;
    for (const input of inputs) {
        const i = slotIndex(input.name);
        if (i >= 0 && input.link != null) lastConnected = Math.max(lastConnected, i);
    }
    let lastLinkedOutput = -1;
    outputs.forEach((output, i) => {
        if (output.links?.length) lastLinkedOutput = i;
    });
    // never drop an output that still feeds something downstream
    const want = Math.min(max, Math.max(1, lastConnected + 2, lastLinkedOutput + 1));

    for (let k = inputs.length - 1; k >= 0; k--) {
        const i = slotIndex(inputs[k].name);
        if (i >= want && inputs[k].link == null) node.removeInput(k);
    }
    for (let i = 0; i < want; i++) {
        const name = slotName(i);
        if (!node.inputs?.some((input) => input.name === name)) node.addInput(name, "IMAGE");
    }

    while ((node.outputs?.length ?? 0) > want &&
           !node.outputs[node.outputs.length - 1].links?.length) {
        node.removeOutput(node.outputs.length - 1);
    }
    while ((node.outputs?.length ?? 0) < want) {
        node.addOutput(slotName(node.outputs?.length ?? 0), "IMAGE");
    }
}
