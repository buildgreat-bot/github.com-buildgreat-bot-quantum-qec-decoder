import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

def verify_diagram(name: str, png_path: str, svg_path: str, min_w: int = 4000):
    print(f"\n--- Verifying {name} Artifacts ---")

    # 1. Verify PNG existence and size
    assert os.path.exists(png_path), f"Missing PNG artifact: {png_path}"
    png_bytes = os.path.getsize(png_path)
    print(f"[+] Found PNG: {png_path} ({png_bytes:,} bytes)")
    assert png_bytes > 100 * 1024, f"PNG artifact too small ({png_bytes} bytes)"

    # 2. Verify PNG image metadata and orientation
    with Image.open(png_path) as img:
        dpi = img.info.get("dpi")
        w, h = img.size
        print(f"    - Dimensions: {w} x {h} px")
        print(f"    - Color Mode: {img.mode}")
        print(f"    - DPI:        {dpi}")
        assert w >= min_w, f"Expected width >= {min_w}, got {w}"
        if dpi:
            assert round(dpi[0]) >= 299, f"Expected ~300 DPI, got {dpi[0]}"

        # Orientation check: top 5% strip contains dark navy header banner (#0b2545)
        # bottom 5% strip contains light background (#f8fafc)
        arr = np.array(img.convert("RGB"))
        top_5pct_mean = float(np.mean(arr[:int(h * 0.05), :, :]))
        bot_5pct_mean = float(np.mean(arr[int(h * 0.95):, :, :]))
        print(f"    - Top 5% Brightness:    {top_5pct_mean:.1f}")
        print(f"    - Bottom 5% Brightness: {bot_5pct_mean:.1f}")
        assert top_5pct_mean < 130.0, f"Expected dark header banner at top, got mean brightness {top_5pct_mean}"
        assert bot_5pct_mean > 200.0, f"Expected light background at bottom, got mean brightness {bot_5pct_mean}"
        assert top_5pct_mean < bot_5pct_mean - 80.0, (
            f"Orientation inverted! Top ({top_5pct_mean}) is not darker than bottom ({bot_5pct_mean})"
        )
        print("    - Orientation: VALID (Header at TOP, content at BOTTOM)")

    # 3. Verify SVG existence and XML validity
    assert os.path.exists(svg_path), f"Missing SVG artifact: {svg_path}"
    svg_bytes = os.path.getsize(svg_path)
    print(f"[+] Found SVG: {svg_path} ({svg_bytes:,} bytes)")
    assert svg_bytes > 50 * 1024, f"SVG artifact too small ({svg_bytes} bytes)"

    tree = ET.parse(svg_path)
    root = tree.getroot()
    assert root.tag.endswith("svg"), f"Root tag must be SVG, got {root.tag}"
    print(f"    - XML Root:   {root.tag}")
    print(f"    - SVG Width:  {root.attrib.get('width')}")
    print(f"    - SVG Height: {root.attrib.get('height')}")

    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    assert r"\u2022" not in svg_content, f"Literal escape \\u2022 found in {svg_path}"
    assert r"\u03bc" not in svg_content, f"Literal escape \\u03bc found in {svg_path}"
    print("    - Unicode:    VALID (No literal escapes)")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(base_dir) == "scripts":
        repo_root = os.path.dirname(base_dir)
    else:
        repo_root = base_dir

    figures_dir = os.path.join(repo_root, "figures")

    print("=================================================================")
    print("             Verifying Publication Diagram Artifacts             ")
    print("=================================================================")

    # Verify System Architecture Diagram
    verify_diagram(
        "System Architecture Diagram",
        os.path.join(figures_dir, "system_architecture.png"),
        os.path.join(figures_dir, "system_architecture.svg"),
        min_w=5000
    )

    # Verify System Workflow Diagram
    verify_diagram(
        "System Workflow Diagram",
        os.path.join(figures_dir, "system_workflow.png"),
        os.path.join(figures_dir, "system_workflow.svg"),
        min_w=4000
    )

    print("\n=================================================================")
    print("        ALL PUBLICATION DIAGRAM ARTIFACTS VERIFIED (PASS)        ")
    print("=================================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
