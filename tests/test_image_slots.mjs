import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../web/js/image_slots.js", import.meta.url), "utf8");
const { syncImageSlots, slotName } = await import(
    `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

function makeNode(inputCount, outputCount) {
    return {
        // widget sockets live in inputs too on the new frontend; they must survive
        inputs: [{ name: "max_megapixels", link: null, widget: {} },
                 ...Array.from({ length: inputCount }, (_, i) => ({ name: slotName(i), link: null }))],
        outputs: Array.from({ length: outputCount }, (_, i) => ({ name: slotName(i), links: null })),
        addInput(name, type) { this.inputs.push({ name, type, link: null }); },
        removeInput(k) { this.inputs.splice(k, 1); },
        addOutput(name, type) { this.outputs.push({ name, type, links: null }); },
        removeOutput(k) { this.outputs.splice(k, 1); },
    };
}
const names = (slots) => slots.filter((s) => !s.widget).map((s) => s.name);
const img = (node, name) => node.inputs.find((s) => s.name === name);

// fresh node: Python declares 16 slots, frontend trims to one
const node = makeNode(16, 16);
syncImageSlots(node, 16);
assert.deepEqual(names(node.inputs), ["image"]);
assert.deepEqual(names(node.outputs), ["image"]);
assert.equal(node.inputs[0].name, "max_megapixels");

// connect "image" -> image_2 appears, on both sides
img(node, "image").link = 1;
syncImageSlots(node, 16);
assert.deepEqual(names(node.inputs), ["image", "image_2"]);
assert.deepEqual(names(node.outputs), ["image", "image_2"]);

// fill image_2 and image_3, then disconnect image_2: a gap is kept, tail trimmed to +1
img(node, "image_2").link = 2;
syncImageSlots(node, 16);
img(node, "image_3").link = 3;
syncImageSlots(node, 16);
assert.deepEqual(names(node.inputs), ["image", "image_2", "image_3", "image_4"]);
img(node, "image_2").link = null;
syncImageSlots(node, 16);
assert.deepEqual(names(node.inputs), ["image", "image_2", "image_3", "image_4"]);
img(node, "image_3").link = null;
syncImageSlots(node, 16);
assert.deepEqual(names(node.inputs), ["image", "image_2"]);
assert.deepEqual(names(node.outputs), ["image", "image_2"]);

// an output still linked downstream is never removed, even if its input is empty
const kept = makeNode(4, 4);
kept.outputs[3].links = [9];
syncImageSlots(kept, 16);
assert.deepEqual(names(kept.inputs), ["image", "image_2", "image_3", "image_4"]);
assert.deepEqual(names(kept.outputs), ["image", "image_2", "image_3", "image_4"]);

// capped at max
const full = makeNode(16, 16);
full.inputs.forEach((s) => { if (!s.widget) s.link = 1; });
syncImageSlots(full, 16);
assert.equal(names(full.inputs).length, 16);
assert.equal(full.outputs.length, 16);

// old single-slot workflow loads and gets its spare slot
const old = makeNode(1, 1);
old.inputs[1].link = 5;
old.outputs[0].links = [6];
syncImageSlots(old, 16);
assert.deepEqual(names(old.inputs), ["image", "image_2"]);
assert.deepEqual(names(old.outputs), ["image", "image_2"]);
console.log("Image slots: grow on connect, trim on disconnect, keep linked outputs, cap at max OK");
