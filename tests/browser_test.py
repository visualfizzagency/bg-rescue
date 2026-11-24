"""Playwright browser test for BG Rescue web tool.
Serves the repo on localhost:8000, loads the page with the LOCAL model dir
(?model= override), adds 5 real photos, waits for the batch, saves outputs.
"""
import asyncio
import base64
import json
import os

ROOT = "/home/hatch/workspace/bg-rescue"
PHOTOS = ["person", "product", "dog", "market", "cat"]


async def main():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/home/hatch/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome",
            args=[
            "--proxy-server=http://127.0.0.1:18080",
            "--ignore-certificate-errors",
            "--disable-dev-shm-usage",
        ])
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: errors.append(f"reqfail: {r.url[:140]} :: {r.failure}"))

        await page.goto(
            "http://localhost:8000/?model=/.test-models/birefnet-lite-512&threads=1",
            timeout=60000,
        )
        await page.wait_for_selector("#dropzone", timeout=30000)
        print("page loaded, title:", await page.title())

        await page.set_input_files(
            "#fileInput",
            [os.path.join(ROOT, "tests/photos", f"{n}.jpg") for n in PHOTOS],
        )
        print("files added, waiting for batch to finish (up to 10 min)...")
        await page.wait_for_function(
            "window.BGRescue && window.BGRescue.items.length===5 && "
            "window.BGRescue.items.every(i=>i.status==='done'||i.status==='error')",
            timeout=600000,
        )
        results = await page.evaluate("window.BGRescue.results")
        print(json.dumps(results, indent=1))

        for i, r in enumerate(results):
            if not r["resultUrl"]:
                continue
            b64 = await page.evaluate(
                """async (url) => {
                    const b = await (await fetch(url)).blob();
                    const buf = await b.arrayBuffer();
                    const bytes = new Uint8Array(buf);
                    let s = '';
                    for (let j = 0; j < bytes.length; j += 8192) {
                        s += String.fromCharCode.apply(null, bytes.subarray(j, j + 8192));
                    }
                    return btoa(s);
                }""",
                r["resultUrl"],
            )
            out = os.path.join(ROOT, "tests/output", f"browser-{PHOTOS[i]}.png")
            with open(out, "wb") as f:
                f.write(base64.b64decode(b64))
            print("saved", out, os.path.getsize(out), "bytes")

        # Download-all (.zip) test
        try:
            async with page.expect_download(timeout=120000) as dl_info:
                await page.click("#downloadAllBtn")
            dl = await dl_info.value
            zpath = os.path.join(ROOT, "tests/output", "browser-all.zip")
            await dl.save_as(zpath)
            print("ZIP OK:", dl.suggested_filename, os.path.getsize(zpath), "bytes")
        except Exception as e:
            print("ZIP FAILED:", e)

        await page.screenshot(path=os.path.join(ROOT, "tests/output", "browser-page.png"), full_page=True)
        print("CONSOLE/PAGE ERRORS:", json.dumps(errors, indent=1)[:4000] or "none")
        await browser.close()


asyncio.run(main())
