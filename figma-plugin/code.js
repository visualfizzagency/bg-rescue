// BG Rescue — Figma plugin (main thread).
// Removes the background from selected images using BiRefNet (MIT),
// running 100% in the plugin UI iframe. No image ever leaves the device.
//
// Flow: main thread exports the selected image bytes -> UI iframe runs the
// AI model (downloaded once from Hugging Face) -> main thread replaces the
// node's image fill with the transparent PNG.

const jobs = new Map(); // id -> { node, fillIndex }
let jobSeq = 0;

figma.showUI(__html__, { width: 320, height: 220, title: 'BG Rescue' });

function imageFills(node) {
  if (!node || !Array.isArray(node.fills)) return [];
  const out = [];
  node.fills.forEach((f, i) => {
    if (f.type === 'IMAGE' && f.imageHash) out.push({ fill: f, index: i });
  });
  return out;
}

function collectTargets() {
  const targets = [];
  for (const node of figma.currentPage.selection) {
    for (const { index } of imageFills(node)) {
      targets.push({ node, fillIndex: index });
    }
  }
  return targets;
}

async function run() {
  const targets = collectTargets();
  if (targets.length === 0) {
    figma.ui.postMessage({ type: 'error', message: 'Select a shape, frame or image with an image fill first, then run BG Rescue again.' });
    return;
  }
  figma.ui.postMessage({ type: 'status', message: `Found ${targets.length} image${targets.length > 1 ? 's' : ''}. Preparing…` });
  for (const t of targets) {
    const id = ++jobSeq;
    jobs.set(id, t);
    try {
      const hash = t.node.fills[t.fillIndex].imageHash;
      const image = figma.getImageByHash(hash);
      if (!image) throw new Error('Could not read the selected image.');
      const bytes = await image.getBytesAsync();
      figma.ui.postMessage({ type: 'process', id, name: t.node.name || 'image', bytes });
      // The UI replies with { type: 'done', id, png } or { type: 'failed', id, error }.
      // Wait for it before moving to the next image (keeps memory low).
      await new Promise((resolve) => {
        const handler = (msg) => {
          if (msg && msg.id === id && (msg.type === 'done' || msg.type === 'failed')) {
            figma.ui.off('message', handler);
            resolve();
          }
        };
        figma.ui.on('message', handler);
      });
    } catch (e) {
      jobs.delete(id);
      figma.notify(`BG Rescue: ${e.message || e}`);
    }
  }
}

figma.ui.on('message', async (msg) => {
  if (!msg) return;
  if (msg.type === 'ready') {
    run();
    return;
  }
  if (msg.type === 'done') {
    const job = jobs.get(msg.id);
    jobs.delete(msg.id);
    if (!job) return;
    try {
      const newImage = figma.createImage(new Uint8Array(msg.png));
      const fills = JSON.parse(JSON.stringify(job.node.fills));
      fills[job.fillIndex] = { ...fills[job.fillIndex], imageHash: newImage.hash };
      job.node.fills = fills;
      figma.notify(`Background removed — ${msg.name || 'image'} is now transparent.`);
    } catch (e) {
      figma.notify(`BG Rescue: could not apply the result (${e.message || e}).`);
    }
    return;
  }
  if (msg.type === 'failed') {
    jobs.delete(msg.id);
    figma.notify(`BG Rescue failed: ${msg.error || 'unknown error'}`);
    return;
  }
  if (msg.type === 'close') {
    figma.closePlugin();
  }
});
