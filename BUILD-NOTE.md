# BG Rescue — build note

**Date:** 24 Nov 2026. **Repo:** https://github.com/visualfizzagency/bg-rescue (public, MIT).
**Why:** remove.bg shuts down 1 Dec 2026. Free replacement, no server.

## Engine (licence-verified)

- Model: `studioludens/birefnet-lite-512` — 512×512 ONNX re-export of
  `ZhengPeng7/BiRefNet_lite`. **MIT** (code and weights), © ZhengPeng7.
  Verified on the Hugging Face model card (`license: mit`) and by multiple
  independent notices (Softery, Alphelo, AbsolTools, deformed.gg).
- The 1024×1024 BiRefNet ONNX variants OOM in browsers — 512 is the
  browser-safe choice (documented by the re-export author and confirmed by
  independent benchmarks).
- **NEVER used:** rembg default BRIA RMBG-2.0 (CC BY-NC 4.0 — not free for
  commercial use).
- Runtime: `@huggingface/transformers@4.3.2` (Apache-2.0) + ONNX Runtime Web.
  WebGPU → fp16 model (94 MB); WASM/CPU → fp32 model (183 MB).
  Input tensor `input_image`, output tensor `output_image` (both confirmed
  by direct inspection of the ONNX graph).

## Web tool (repo root)

- `index.html` + `assets/style.css` + `assets/app.js`. Static, no build step.
- Batch up to 20, full-res output (longest side capped at 4096 px),
  per-image PNG download + Download-all (.zip via JSZip, lazy-loaded).
- Before/after: hold-to-see-original on each card; checkerboard preview.
- Privacy: model downloads once from Hugging Face; photos never leave the device.
- Dev overrides (not for users): `?model=<local model dir>` and `?threads=<n>`.
- `window.BGRescue` debug handle exposes items/results (used by the test).

## Figma plugin (`figma-plugin/`)

- `manifest.json`, `code.js` (main thread: export image bytes, swap fill),
  `ui.html` (same engine in the UI iframe), `README.md`.
- Validated: manifest parses, `code.js` + inline UI script pass `node --check`.
- **NOT verified end-to-end in Figma** (no Figma on this machine) — the
  README lists the 5 manual checks needed before publishing.

## Test results

### Python/ONNX sanity check (native onnxruntime, 24 Nov 2026)
- Model loads; `input_image` [1,3,512,512] fp32 in, `output_image` [1,1,512,512] out.
- 5 real photos, inference 4.1–7.3 s each on this VM's CPU:
  - person (Pexels portrait): keep 25% / cut 74% — clean cutout, hair edges good.
  - product (red sneakers, CC0): keep 20% / cut 80%.
  - dog (golden retriever): keep 9% / cut 90% — clean fur edges; thin leash
    mostly cut (small stub at the collar remains — expected for a 1-px line).
  - market (busy street): keep 21% / cut 78%.
  - cat: keep 21% / cut 78%.
- All 5 outputs verified as real transparent PNGs (alpha channel checked;
  checkerboard composites inspected visually).

### Browser test (Playwright/Chromium, local server, local model)
- Test: `tests/browser_test.py` — page served on localhost:8000, model loaded
  from the local mirror (`?model=/.test-models/birefnet-lite-512&threads=1`),
  transformers.js loaded from jsDelivr CDN through the proxy.
- **PASSED, 24 Nov 2026, total 91 s for model load + 5 images:**
  - Page loads, correct title, no missing assets.
  - All 5 photos status `done` at full resolution (1024×683, 768×1024).
  - 5 real transparent PNGs saved (137–427 KB each).
  - Browser masks match the Python reference almost exactly
    (mean abs alpha diff 0.17–0.38 / 255 — float rounding only).
  - **Download-all (.zip) works:** `bg-rescue-output.zip`, 1.6 MB, valid,
    contains all 5 `*-no-bg.png` files.
  - **Zero console errors, zero page errors, zero failed requests.**
  - Screenshot (`tests/output/browser-page.png`) inspected: clean professional
    layout — hero, dropzone, status bar, 5 cards with checkerboard previews,
    Done badges, sizes, download buttons, How-it-works, Why-free, footer.
- Cutout quality (from screenshot): person hair clean, sneaker sole lights kept,
  dog fur clean, market couple kept as the salient subjects, cat clean.

## Known limits (honest)

- First visit downloads the model (94 MB WebGPU / 183 MB CPU) — one time,
  cached afterwards. Slow connections will wait; progress is shown.
- 512×512 inference: very fine detail (single hairs, thin wires, leash lines)
  can be softened or partly cut. Full-res mask upscale keeps edges sharp.
- Output longest side capped at 4096 px (browser memory guard).
- Very large batches on low-memory devices: images process one at a time,
  but each keeps its result in memory until downloaded/cleared.
- The Figma plugin UI iframe must reach cdn.jsdelivr.net + huggingface.co;
  offline Figma use will not load the model.
