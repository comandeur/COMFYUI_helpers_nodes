import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../web/js/animation_inputs.js", import.meta.url), "utf8");
const { restoreAnimationNumbers } = await import(
    `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const defaults = {
    force_rate: 0, custom_width: 0, custom_height: 0,
    frame_load_cap: 0, skip_first_frames: 0, select_every_nth: 1,
};
const inputs = Object.fromEntries(Object.entries(defaults).map(([name, value]) =>
    [name, [name === "force_rate" ? "FLOAT" : "INT", { default: value }]]));
inputs.image = [["test.gif"], {}];
for (const value of [null, undefined, NaN, Infinity]) {
    const node = { widgets: Object.keys(defaults).map((name) => ({ name, value })) };
    restoreAnimationNumbers(node, inputs);
    assert.deepEqual(Object.fromEntries(node.widgets.map((w) => [w.name, w.value])), defaults);
    assert.deepEqual(JSON.parse(JSON.stringify(node.widgets)), node.widgets);
}
const node = {
    widgets: [{ name: "force_rate", value: 23.976 }, { name: "custom_width", value: 720 },
        { name: "custom_height", value: null }, { name: "image", value: "test.gif" }],
    inputs: [{ name: "custom_height", link: 0 }],
};
const original = structuredClone(node);
restoreAnimationNumbers(node, inputs);
assert.deepEqual(node, original);
node.inputs[0].link = null;
restoreAnimationNumbers(node, inputs);
assert.equal(node.widgets[2].value, 0);
console.log("Animation numeric recovery: six null inputs, non-finite values, saved values and links OK");
