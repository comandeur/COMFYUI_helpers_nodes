import { app } from "../../../scripts/app.js";
import { syncImageSlots } from "./image_slots.js";

const NODE = "CMDR_LimitImageMegapixels";
const MAX_IMAGES = 16; // keep in sync with MAX_IMAGES in helpers_nodes/image_nodes.py

function chain(object, property, callback) {
    const original = object[property];
    object[property] = function () {
        const r = original?.apply(this, arguments);
        callback.apply(this, arguments);
        return r;
    };
}

function scheduleSync(node) {
    if (node.__cmdrSyncPending) return;
    node.__cmdrSyncPending = true;
    // defer: links are still being wired while connect/configure callbacks run
    setTimeout(() => {
        node.__cmdrSyncPending = false;
        if (app.configuringGraph || !node.graph) {
            scheduleSync(node);
            return;
        }
        syncImageSlots(node, MAX_IMAGES);
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], size[0]), size[1]]);
        node.setDirtyCanvas?.(true, true);
    }, 0);
}

app.registerExtension({
    name: "CMDR.LimitImageMegapixels",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;
        chain(nodeType.prototype, "onNodeCreated", function () { scheduleSync(this); });
        chain(nodeType.prototype, "onConfigure", function () { scheduleSync(this); });
        chain(nodeType.prototype, "onConnectionsChange", function () { scheduleSync(this); });
    },
});
