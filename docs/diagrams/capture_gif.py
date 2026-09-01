"""
Capture animated HTML diagrams as GIF files for GitHub-native rendering.

Uses Playwright to open each HTML file, captures frames across one
animation cycle, and stitches them into an optimized animated GIF
via Pillow and ffmpeg/gifsicle.

Usage:
    python3 capture_gif.py
"""

import os
import time
import subprocess
from playwright.sync_api import sync_playwright
from PIL import Image

DIAGRAMS_DIR = os.path.dirname(__file__)

# Each entry: (html_filename, output_gif, viewport_width, viewport_height)
TARGETS = [
    (
        "deployment_topology_animated.html",
        "deployment_topology_animated.gif",
        1200,
        900,
    ),
    ("architecture_animated.html", "architecture_animated.gif", 1250, 850),
]

# Animation settings
FRAME_COUNT = 20  # frames per cycle
CYCLE_DURATION = 1.2  # seconds (matches CSS animation duration)
FRAME_DELAY_MS = int(CYCLE_DURATION / FRAME_COUNT * 1000)  # ~60ms per frame
GIF_FRAME_DURATION = FRAME_DELAY_MS  # match real-time speed


def capture_frames(page, html_path, num_frames, delay_ms, viewport):
    """Capture sequential screenshots of an animated HTML page."""
    width, height = viewport
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"file://{html_path}")
    page.wait_for_load_state("networkidle")
    # Let the first animation cycle start
    time.sleep(0.3)

    frames = []
    for i in range(num_frames):
        screenshot = page.screenshot(type="png")
        frames.append(screenshot)
        if i < num_frames - 1:
            time.sleep(delay_ms / 1000.0)

    return frames


def frames_to_gif(frame_bytes_list, output_path, duration_ms):
    """Convert a list of PNG byte arrays to an animated GIF."""
    images = []
    for frame_bytes in frame_bytes_list:
        # Write temp PNG and read with Pillow
        import io

        img = Image.open(io.BytesIO(frame_bytes))
        # Convert to RGBA then to palette mode for GIF
        img = img.convert("RGBA")
        images.append(img)

    # Save animated GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,  # infinite loop
        optimize=False,
    )


def optimize_gif(gif_path):
    """Optimize GIF size with gifsicle if available."""
    try:
        optimized = gif_path.replace(".gif", "_opt.gif")
        subprocess.run(
            [
                "gifsicle",
                "-O3",
                "--lossy=80",
                "--colors=128",
                gif_path,
                "-o",
                optimized,
            ],
            check=True,
            capture_output=True,
        )
        os.replace(optimized, gif_path)
        print("  Optimized with gifsicle")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  gifsicle not available, skipping optimization")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=2)  # 2x for retina clarity

        for html_file, gif_file, vw, vh in TARGETS:
            html_path = os.path.join(DIAGRAMS_DIR, html_file)
            gif_path = os.path.join(DIAGRAMS_DIR, gif_file)

            if not os.path.exists(html_path):
                print(f"Skipping {html_file} (not found)")
                continue

            print(f"Capturing {html_file} ...")
            frames = capture_frames(
                page, html_path, FRAME_COUNT, FRAME_DELAY_MS, (vw, vh)
            )
            print(f"  Captured {len(frames)} frames")

            frames_to_gif(frames, gif_path, GIF_FRAME_DURATION)
            raw_size = os.path.getsize(gif_path) / 1024
            print(f"  Raw GIF: {raw_size:.0f} KB")

            optimize_gif(gif_path)
            final_size = os.path.getsize(gif_path) / 1024
            print(f"  Final: {gif_path} ({final_size:.0f} KB)")

        browser.close()


if __name__ == "__main__":
    main()
