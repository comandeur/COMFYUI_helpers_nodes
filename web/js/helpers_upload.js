import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// The built-in image upload widget only offers png/jpeg/webp in the file
// dialog, so a .gif can't be picked. We add our own button + drop target with
// an animation-aware accept list, plus an <img> preview (an <img> plays gif and
// animated webp natively, no player needed).

const TARGET_NODES = ["Helpers_LoadAnimationUpload"];
const ACCEPT_MIME = ["image/gif", "image/webp", "image/apng", "image/png"];
const ACCEPT = ".gif,.webp,.apng,image/gif,image/webp,image/apng";

function chainCallback(object, property, callback) {
    if (object == undefined) {
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property];
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r;
        };
    } else {
        object[property] = callback;
    }
}

function isAnimationFile(file) {
    if (!file) {
        return false;
    }
    if (ACCEPT_MIME.includes(file.type)) {
        return true;
    }
    return /\.(gif|webp|apng)$/i.test(file.name ?? "");
}

async function uploadAnimation(file) {
    const body = new FormData();
    body.append("image", file);
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (resp.status !== 200) {
        alert(`Upload failed: ${resp.status} ${resp.statusText}`);
        return null;
    }
    const data = await resp.json();
    return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function addUploadWidget(nodeType, widgetName = "image") {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const node = this;
        const fileWidget = this.widgets.find((w) => w.name === widgetName);
        if (!fileWidget) {
            return;
        }

        const applyUpload = async (file) => {
            const path = await uploadAnimation(file);
            if (!path) {
                return false;
            }
            if (!fileWidget.options.values.includes(path)) {
                fileWidget.options.values.push(path);
            }
            fileWidget.value = path;
            fileWidget.callback?.(path);
            return true;
        };

        const fileInput = document.createElement("input");
        Object.assign(fileInput, {
            type: "file",
            accept: ACCEPT,
            style: "display: none",
            onchange: async () => {
                if (fileInput.files.length) {
                    await applyUpload(fileInput.files[0]);
                }
                fileInput.value = "";
            },
        });
        document.body.append(fileInput);
        chainCallback(this, "onRemoved", () => fileInput.remove());

        this.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
        this.onDragDrop = async function (e) {
            const file = e?.dataTransfer?.files?.[0];
            if (!isAnimationFile(file)) {
                return false;
            }
            return await applyUpload(file);
        };

        const uploadWidget = this.addWidget("button", "choose animation to upload", "image", () => {
            // clear the active click event so the canvas doesn't swallow it
            app.canvas.node_widget = null;
            fileInput.click();
        });
        uploadWidget.options.serialize = false;
        node.setDirtyCanvas(true, true);
    });
}

function addAnimationPreview(nodeType, widgetName = "image") {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const previewNode = this;
        const container = document.createElement("div");
        container.style.width = "100%";

        const imgEl = document.createElement("img");
        imgEl.style.width = "100%";
        imgEl.style.display = "block";
        container.appendChild(imgEl);

        const previewWidget = this.addDOMWidget("animationpreview", "preview", container, {
            serialize: false,
            hideOnZoom: false,
        });
        previewWidget.computeSize = function (width) {
            if (this.aspectRatio) {
                let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4]; // nothing loaded: collapse the widget
        };

        imgEl.addEventListener("load", () => {
            if (imgEl.naturalHeight) {
                previewWidget.aspectRatio = imgEl.naturalWidth / imgEl.naturalHeight;
                previewNode.setSize([previewNode.size[0],
                                     previewNode.computeSize([previewNode.size[0], previewNode.size[1]])[1]]);
                previewNode.graph?.setDirtyCanvas(true);
            }
        });
        imgEl.addEventListener("error", () => {
            previewWidget.aspectRatio = undefined;
            previewNode.graph?.setDirtyCanvas(true);
        });
        // let a drop anywhere over the preview reach the node handler
        container.addEventListener("dragover", (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            app.dragOverNode = previewNode;
        });

        const fileWidget = this.widgets.find((w) => w.name === widgetName);
        if (!fileWidget) {
            return;
        }
        const showPreview = (value) => {
            if (!value) {
                imgEl.removeAttribute("src");
                previewWidget.aspectRatio = undefined;
                return;
            }
            const clean = String(value).replace(/\s*\[[^\]]+\]\s*$/, "");
            const idx = clean.lastIndexOf("/");
            const params = new URLSearchParams({
                filename: idx >= 0 ? clean.slice(idx + 1) : clean,
                subfolder: idx >= 0 ? clean.slice(0, idx) : "",
                type: "input",
            });
            imgEl.src = api.apiURL(`/view?${params}`);
        };
        chainCallback(fileWidget, "callback", showPreview);
        requestAnimationFrame(() => showPreview(fileWidget.value));
    });
}

app.registerExtension({
    name: "Helpers.AnimationUpload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODES.includes(nodeData?.name)) {
            return;
        }
        addUploadWidget(nodeType);
        addAnimationPreview(nodeType);
    },
});
