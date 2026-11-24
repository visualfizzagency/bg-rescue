# BG Rescue — Figma plugin

Remove the background from any selected image in Figma, using the same free
BiRefNet (MIT) engine as the [BG Rescue web tool](../README.md). The AI model
downloads once to the plugin and everything runs on your device — no uploads,
no account, no cost.

## Install (development plugin)

1. In Figma: **Plugins → Development → Import plugin from manifest…**
2. Pick `manifest.json` from this folder.
3. The plugin appears under **Plugins → Development → BG Rescue**.

## Use

1. Select a shape, frame or image that has an **image fill**.
2. Run **BG Rescue**.
3. The image fill is replaced with the transparent PNG (same size, same node).
   The original image is not kept — duplicate the node first if you want a copy.

## Publish (when ready)

1. **Plugins → Development → Publish** — needs a Figma account; community
   plugins go through Figma's review.
2. Name: **BG Rescue**, category: image editing / productivity.
3. The plugin is free — never charge for it. The support link in the web
   tool's README is the only monetisation (donations).

## How it works

- `code.js` (main thread) exports the selected image bytes and swaps the fill.
- `ui.html` (UI iframe) loads `@huggingface/transformers` from CDN, downloads
  the `studioludens/birefnet-lite-512` model (BiRefNet, MIT) from Hugging Face
  once, runs inference at 512×512, and applies the mask at full resolution.
- GPU (WebGPU) is used when available, otherwise CPU (WASM).
- The UI iframe needs network access to `cdn.jsdelivr.net` and
  `huggingface.co` (declared in `manifest.json` under `networkAccess`).

## What is NOT verified yet

There is no Figma on this build machine, so the plugin is **not yet tested
end-to-end in Figma**. Before publishing, manually check in Figma:

1. The plugin imports from the manifest without errors.
2. Selecting an image and running it downloads the model (progress shows).
3. The fill is replaced with a transparent PNG (checkerboard behind it).
4. A second image in the same session reuses the cached model (no re-download).
5. The plugin behaves on a shape with an image fill (not just a bare image node).

`code.js` passes `node --check` and `manifest.json` is valid JSON — the
remaining checks above are the only unverified part.
