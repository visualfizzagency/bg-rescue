# BG Rescue

**The free remove.bg alternative — forever free, runs 100% in your browser.**

Remove the background from any photo: drag in up to 20 photos at once, get full-resolution transparent PNGs back. No upload, no account, no watermark, no credits. Your photos never leave your device.

remove.bg shuts down on 1 December 2026. BG Rescue is built to stay free — there is no server to pay for, so there is nothing to shut down.

## Try it

**Live site:** https://visualfizzagency.github.io/bg-rescue/ *(GitHub Pages — enable in repo settings after cloning, or host anywhere static)*

Or run it locally — it is a plain static site:

```bash
git clone https://github.com/visualfizzagency/bg-rescue.git
cd bg-rescue
python3 -m http.server 8000
# open http://localhost:8000
```

## What it does

- **Batch** — up to 20 photos at a time, processed one after another
- **Full resolution** — output keeps the original photo size (longest side capped at 4096 px so the browser stays responsive)
- **Transparent PNG** — per-photo download, or all at once as a `.zip`
- **Before / after** — hold the "original" button on any result to compare
- **Samples** — try it instantly with the two built-in sample photos
- **Private by design** — the AI model downloads once to your browser; every photo is processed on your own device. No analytics, no tracking, no cookies.

## How it works

The background removal runs on **BiRefNet** (Zheng et al., 2024) — the state-of-the-art dichotomous image segmentation model — in the browser-ready 512×512 ONNX export, via [ONNX Runtime Web](https://onnxruntime.ai/) and [@huggingface/transformers](https://huggingface.co/docs/transformers.js). Inference runs at 512×512, then the alpha mask is upscaled and applied at full resolution so edges stay sharp.

- **Model:** [`studioludens/birefnet-lite-512`](https://huggingface.co/studioludens/birefnet-lite-512) — a 512×512 ONNX re-export of `ZhengPeng7/BiRefNet_lite`
- **Model licence:** MIT (code and weights), © ZhengPeng7. See [Credits](#credits) below.
- **First load:** the model downloads once (~94 MB on WebGPU, ~183 MB on CPU/WASM) and is cached by the browser afterwards.
- **Engine:** GPU (WebGPU) when available, otherwise CPU (WASM). No server, no API key, no cost per image.

## Why it's free

There is no server. The model lives in your browser, the work happens on your device, and the code is MIT-licensed. Nothing to pay for, nothing to sell you.

## Figma plugin

A Figma plugin using the same engine lives in [`figma-plugin/`](figma-plugin/) — select an image, remove its background, get a transparent PNG back. See [`figma-plugin/README.md`](figma-plugin/README.md) for install and testing notes.

## Credits

- **BiRefNet** by Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, Nicu Sebe — [GitHub](https://github.com/ZhengPeng7/BiRefNet) / [Hugging Face](https://huggingface.co/ZhengPeng7/BiRefNet_lite) — **MIT License, Copyright (c) 2024 ZhengPeng**. The 512×512 ONNX export used here is by [studioludens](https://huggingface.co/studioludens/birefnet-lite-512), an unmodified re-export of the MIT-licensed weights.
- **ONNX Runtime Web** by Microsoft — MIT License.
- **@huggingface/transformers** — Apache License 2.0.
- **JSZip** — MIT License.
- **Sample photos:** person — Pexels (free licence); product — Rawpixel, CC0.

### What this is NOT

This tool does **not** use `rembg`'s default model. The BRIA RMBG-2.0 model that ships as rembg's default is **CC BY-NC 4.0 — not free for commercial use** — so it is deliberately not used here. BG Rescue uses BiRefNet (MIT) only.

## Support

BG Rescue is free forever. If it saves you time, you can support the project here:

**https://www.supportkori.com/hellomdayub**

Built by [Md Ayub Mondol](https://mdayubmondol.com) / VisualFizz, Dhaka.

## License

MIT — see [LICENSE](LICENSE).
