/* BG Rescue — free batch background remover.
 * Engine: BiRefNet (MIT) via @huggingface/transformers, ONNX Runtime Web.
 * Everything runs in the browser. Photos are never uploaded anywhere.
 */
'use strict';

const MODEL_ID = 'studioludens/birefnet-lite-512'; // BiRefNet-lite 512px ONNX re-export, MIT
const TRANSFORMERS_URLS = [
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js',
  'https://unpkg.com/@huggingface/transformers@4.3.2/dist/transformers.min.js',
];
const JSZIP_URL = 'https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js';
const INFER_SIZE = 512;          // model input resolution
const MAX_FILES = 20;            // per batch
const MAX_SIDE = 4096;           // output longest side cap (browser memory guard)
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const params = new URLSearchParams(location.search);
const LOCAL_MODEL = params.get('model') || '';   // dev override: local model dir, e.g. ?model=/models/birefnet-lite-512
const THREADS = parseInt(params.get('threads') || '', 10) || 0;

const $ = (s) => document.querySelector(s);
const dropzone = $('#dropzone');
const fileInput = $('#fileInput');
const grid = $('#grid');
const statusbar = $('#statusbar');
const statusText = $('#statusText');
const progressWrap = $('#progressWrap');
const progressFill = $('#progressFill');
const toolbar = $('#toolbar');
const countLabel = $('#countLabel');
const emptyNote = $('#emptyNote');

let transformers = null;   // the imported @huggingface/transformers module
let model = null;          // loaded AutoModel
let modelLoading = null;   // in-flight load promise
const items = [];          // {id, name, file, origUrl, status, resultBlob, resultUrl, w, h, error}
let seq = 0;

function setStatus(text, progress) {
  statusbar.hidden = false;
  statusText.textContent = text;
  if (typeof progress === 'number') {
    progressWrap.hidden = false;
    progressFill.style.width = Math.max(0, Math.min(100, progress)) + '%';
  } else {
    progressWrap.hidden = true;
  }
}

function updateToolbar() {
  const done = items.filter((i) => i.status === 'done').length;
  toolbar.hidden = items.length === 0;
  emptyNote.style.display = items.length === 0 ? '' : 'none';
  countLabel.textContent = items.length
    ? `${items.length} photo${items.length > 1 ? 's' : ''} — ${done} done`
    : '';
}

async function loadTransformers() {
  if (transformers) return transformers;
  let lastErr = null;
  for (const url of TRANSFORMERS_URLS) {
    try {
      transformers = await import(url);
      return transformers;
    } catch (e) { lastErr = e; }
  }
  throw new Error('Could not load the AI library (tried jsDelivr and unpkg). Check your connection and reload. ' + (lastErr && lastErr.message ? lastErr.message : ''));
}

async function loadModel() {
  if (model) return model;
  if (modelLoading) return modelLoading;
  modelLoading = (async () => {
    const tf = await loadTransformers();
    const { AutoModel, env } = tf;
    if (THREADS > 0) env.backends.onnx.wasm.numThreads = THREADS;
    if (LOCAL_MODEL) env.allowLocalModels = true;
    const modelId = LOCAL_MODEL || MODEL_ID;
    const useWebGPU = !!navigator.gpu;
    const attempts = useWebGPU
      ? [{ device: 'webgpu', dtype: 'fp16' }, { device: 'wasm', dtype: 'fp32' }]
      : [{ device: 'wasm', dtype: 'fp32' }];
    let lastErr = null;
    for (const opts of attempts) {
      try {
        setStatus(`Downloading AI model (${opts.dtype === 'fp16' ? '94 MB' : '183 MB'}, one time only — cached afterwards)…`, 0);
        model = await AutoModel.from_pretrained(modelId, {
          dtype: opts.dtype,
          device: opts.device,
          progress_callback: (p) => {
            if (p.status === 'progress' && p.file && p.file.includes('model') && p.total) {
              setStatus(`Downloading AI model… ${Math.round((p.loaded / p.total) * 100)}% (${opts.dtype === 'fp16' ? '94 MB' : '183 MB'}, one time only)`, (p.loaded / p.total) * 100);
            } else if (p.status === 'progress') {
              setStatus('Downloading AI model…', undefined);
            }
          },
        });
        setStatus(`AI model ready (${opts.device === 'webgpu' ? 'GPU' : 'CPU'} mode).`);
        return model;
      } catch (e) { lastErr = e; }
    }
    throw new Error('Could not load the AI model. ' + (lastErr && lastErr.message ? lastErr.message : ''));
  })();
  try { return await modelLoading; }
  finally { modelLoading = null; }
}

/* --- image helpers --- */
function loadImage(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Could not read this image file.')); };
    img.src = url;
  });
}

function preprocess(img) {
  // Stretch to 512x512, ImageNet-normalise into a [1,3,512,512] float32 tensor.
  const c = document.createElement('canvas');
  c.width = INFER_SIZE; c.height = INFER_SIZE;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, INFER_SIZE, INFER_SIZE);
  const px = ctx.getImageData(0, 0, INFER_SIZE, INFER_SIZE).data;
  const data = new Float32Array(3 * INFER_SIZE * INFER_SIZE);
  for (let i = 0; i < INFER_SIZE * INFER_SIZE; i++) {
    data[i] = (px[i * 4] / 255 - MEAN[0]) / STD[0];
    data[INFER_SIZE * INFER_SIZE + i] = (px[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    data[2 * INFER_SIZE * INFER_SIZE + i] = (px[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new transformers.Tensor('float32', data, [1, 3, INFER_SIZE, INFER_SIZE]);
}

function maskToCanvas(logits) {
  // logits: Float32Array 512*512 -> grayscale mask canvas (white = keep).
  const c = document.createElement('canvas');
  c.width = INFER_SIZE; c.height = INFER_SIZE;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const img = ctx.createImageData(INFER_SIZE, INFER_SIZE);
  for (let i = 0; i < logits.length; i++) {
    const s = 1 / (1 + Math.exp(-logits[i])); // sigmoid
    const v = Math.round(s * 255);
    img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

async function removeBackground(file) {
  const img = await loadImage(file);
  // Cap the working size so giant phone photos don't blow up browser memory.
  const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));

  const mdl = await loadModel();
  const tensor = preprocess(img);
  const out = await mdl({ input_image: tensor });
  const logitsTensor = out.output_image || out.logits || Object.values(out)[0];
  const logits = logitsTensor.data; // Float32Array, 512*512
  const maskSmall = maskToCanvas(logits);

  // Composite at full (capped) resolution: original RGB + upscaled mask as alpha.
  const outCanvas = document.createElement('canvas');
  outCanvas.width = w; outCanvas.height = h;
  const octx = outCanvas.getContext('2d', { willReadFrequently: true });
  octx.drawImage(img, 0, 0, w, h);
  const imgData = octx.getImageData(0, 0, w, h);

  const maskFull = document.createElement('canvas');
  maskFull.width = w; maskFull.height = h;
  const mctx = maskFull.getContext('2d', { willReadFrequently: true });
  mctx.imageSmoothingEnabled = true;
  mctx.imageSmoothingQuality = 'high';
  mctx.drawImage(maskSmall, 0, 0, w, h);
  const maskData = mctx.getImageData(0, 0, w, h).data;

  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    const a = maskData[i]; // red channel of the grayscale mask
    d[i + 3] = a < 5 ? 0 : a; // drop near-invisible noise
  }
  octx.putImageData(imgData, 0, 0);

  const blob = await new Promise((res) => outCanvas.toBlob(res, 'image/png'));
  return { blob, w, h };
}

/* --- UI: cards --- */
const cardTemplate = $('#cardTemplate');

function addCard(item) {
  const node = cardTemplate.content.cloneNode(true);
  const card = node.querySelector('.card');
  card.dataset.id = item.id;
  const img = node.querySelector('.thumb-img');
  img.src = item.origUrl;
  img.alt = item.name;
  node.querySelector('.card-name').textContent = item.name;
  node.querySelector('.card-meta').textContent = 'Queued…';
  node.querySelector('.thumb-badge').textContent = 'Queued';
  const holdBtn = node.querySelector('.hold-btn');
  holdBtn.disabled = true;
  const dlBtn = node.querySelector('.dl-btn');
  dlBtn.addEventListener('click', () => downloadOne(item));
  holdBtn.addEventListener('pointerdown', () => { img.src = item.origUrl; });
  holdBtn.addEventListener('pointerup', () => { if (item.resultUrl) img.src = item.resultUrl; });
  holdBtn.addEventListener('pointerleave', () => { if (item.resultUrl) img.src = item.resultUrl; });
  node.querySelector('.rm-btn').addEventListener('click', () => removeItem(item.id));
  grid.appendChild(node);
  item.el = grid.querySelector(`.card[data-id="${item.id}"]`);
  updateToolbar();
}

function setCard(item, status, meta) {
  if (!item.el) return;
  const badge = item.el.querySelector('.thumb-badge');
  const metaEl = item.el.querySelector('.card-meta');
  const dlBtn = item.el.querySelector('.dl-btn');
  const holdBtn = item.el.querySelector('.hold-btn');
  badge.textContent = status === 'done' ? 'Done' : status === 'error' ? 'Failed' : status === 'processing' ? 'Working…' : 'Queued';
  badge.className = 'thumb-badge' + (status === 'done' ? ' done' : status === 'error' ? ' error' : '');
  if (meta !== undefined) metaEl.textContent = meta;
  if (status === 'done') {
    dlBtn.disabled = false;
    holdBtn.disabled = false;
    item.el.querySelector('.thumb-img').src = item.resultUrl;
  }
}

function removeItem(id) {
  const i = items.findIndex((x) => x.id === id);
  if (i < 0) return;
  const [item] = items.splice(i, 1);
  if (item.el) item.el.remove();
  if (item.origUrl) URL.revokeObjectURL(item.origUrl);
  if (item.resultUrl) URL.revokeObjectURL(item.resultUrl);
  updateToolbar();
}

function downloadOne(item) {
  if (!item.resultBlob) return;
  const a = document.createElement('a');
  a.href = item.resultUrl;
  a.download = item.name.replace(/\.[^.]+$/, '') + '-no-bg.png';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* --- queue --- */
let processing = false;

async function processQueue() {
  if (processing) return;
  processing = true;
  try {
    for (const item of items) {
      if (item.status !== 'queued') continue;
      item.status = 'processing';
      setCard(item, 'processing', 'Removing background…');
      setStatus(`Working on ${item.name}…`);
      try {
        const { blob, w, h } = await removeBackground(item.file);
        item.resultBlob = blob;
        item.resultUrl = URL.createObjectURL(blob);
        item.w = w; item.h = h;
        item.status = 'done';
        setCard(item, 'done', `${w}×${h} · transparent PNG · ${(blob.size / 1024).toFixed(0)} KB`);
      } catch (e) {
        item.status = 'error';
        item.error = e.message || String(e);
        setCard(item, 'error', 'Failed: ' + item.error);
      }
      updateToolbar();
    }
    const done = items.filter((i) => i.status === 'done').length;
    const failed = items.filter((i) => i.status === 'error').length;
    setStatus(failed ? `${done} done, ${failed} failed.` : `All done — ${done} photo${done === 1 ? '' : 's'} ready. Download them below.`);
  } finally {
    processing = false;
  }
}

async function addFiles(fileList) {
  const files = [...fileList].filter((f) => f && f.type && f.type.startsWith('image/'));
  if (!files.length) { setStatus('That was not an image file — try a JPG, PNG or WebP.'); return; }
  const room = MAX_FILES - items.length;
  if (room <= 0) { setStatus(`Batch is full (${MAX_FILES} photos max). Clear some to add more.`); return; }
  const take = files.slice(0, room);
  if (take.length < files.length) setStatus(`Added ${take.length} photos (batch limit is ${MAX_FILES}).`);
  for (const file of take) {
    const item = { id: ++seq, name: file.name || 'photo', file, origUrl: URL.createObjectURL(file), status: 'queued' };
    items.push(item);
    addCard(item);
  }
  updateToolbar();
  processQueue().catch((e) => setStatus('Something went wrong: ' + (e.message || e)));
}

async function downloadAll() {
  const done = items.filter((i) => i.status === 'done');
  if (!done.length) return;
  setStatus('Preparing your .zip…');
  try {
    if (!window.JSZip) {
      await new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = JSZIP_URL;
        s.onload = resolve;
        s.onerror = () => reject(new Error('Could not load the zip library.'));
        document.head.appendChild(s);
      });
    }
    const zip = new window.JSZip();
    for (const item of done) {
      zip.file(item.name.replace(/\.[^.]+$/, '') + '-no-bg.png', item.resultBlob);
    }
    const blob = await zip.generateAsync({ type: 'blob' }, (meta) => {
      setStatus(`Preparing your .zip… ${Math.round(meta.percent)}%`, meta.percent);
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'bg-rescue-output.zip';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setStatus(`Downloaded ${done.length} PNGs as bg-rescue-output.zip.`);
  } catch (e) {
    setStatus('Zip failed (' + (e.message || e) + ') — downloading files one by one instead.');
    for (const item of done) downloadOne(item);
  }
}

/* --- events --- */
dropzone.addEventListener('click', (e) => {
  if (e.target.closest('.sample-btn') || e.target.id === 'browseBtn') return;
  fileInput.click();
});
$('#browseBtn').addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });
['dragenter', 'dragover'].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove('dragging'); }));
dropzone.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));
dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
document.addEventListener('paste', (e) => {
  const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith('image/'));
  if (files.length) addFiles(files);
});
document.querySelectorAll('.sample-btn').forEach((btn) => {
  btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    setStatus('Loading sample photo…');
    try {
      const res = await fetch(btn.dataset.sample);
      if (!res.ok) throw new Error('Sample failed to load.');
      const blob = await res.blob();
      addFiles([new File([blob], btn.dataset.name, { type: blob.type || 'image/jpeg' })]);
    } catch (err) { setStatus('Sample failed to load: ' + (err.message || err)); }
  });
});
$('#clearBtn').addEventListener('click', () => {
  [...items].forEach((i) => removeItem(i.id));
  setStatus('Cleared.');
});
$('#downloadAllBtn').addEventListener('click', downloadAll);

/* Debug/test handle — also used by the automated test. */
window.BGRescue = {
  items,
  get results() {
    return items.map((i) => ({ name: i.name, status: i.status, w: i.w, h: i.h, resultUrl: i.resultUrl || null, error: i.error || null }));
  },
};
