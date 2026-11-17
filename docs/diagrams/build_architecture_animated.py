"""
Build the animated MLServer Software Architecture diagram as a self-contained HTML file.

Shows the internal component perspective at frontend_workers > 1: N frontend
processes (REST, gRPC, metrics, DataPlane, middleware, dispatcher) on
SO_REUSEPORT, the supervisor process owning the shared inference pool and
coordinating model lifecycle over control pipes, and the shared worker
processes (with the adaptive batcher and codec pipeline living next to the
models) reached over the ZeroMQ data plane. At frontend_workers = 1 the
supervisor and the single frontend collapse into one main process.
Complements the deployment topology diagram which shows the K8s
infrastructure perspective.

Uses Kubernetes-style icons from the diagrams library (Apache-2.0 licensed).
All icons are embedded as base64 data URIs so the HTML is fully self-contained.

Usage:
    python3 build_architecture_animated.py
"""

import base64
import os

ICONS_DIR = os.path.join(os.path.dirname(__file__), "icons")
OUT_FILE = os.path.join(os.path.dirname(__file__), "architecture_animated.html")


def b64_img(filename):
    """Read a PNG file and return a base64 data URI."""
    path = os.path.join(ICONS_DIR, filename)
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# Load icons as data URIs
icons = {
    "pod": b64_img("pod.png"),
    "svc": b64_img("svc.png"),
    "deploy": b64_img("deploy.png"),
    "cm": b64_img("cm.png"),
    "pvc": b64_img("pvc.png"),
    "prometheus": b64_img("prometheus.png"),
    "users": b64_img("users.png"),
    "storage": b64_img("storage.png"),
}


def node(icon_key, label, x, y, size=48, sublabel=None, label_color="#333"):
    """Generate an SVG node: icon image + label below."""
    img_x = x - size / 2
    img_y = y - size / 2
    parts = [
        f'  <image href="{icons[icon_key]}" x="{img_x}" y="{img_y}" '
        f'width="{size}" height="{size}"/>',
        f'  <text x="{x}" y="{y + size/2 + 14}" '
        f'class="node-label" fill="{label_color}">{label}</text>',
    ]
    if sublabel:
        parts.append(
            f'  <text x="{x}" y="{y + size/2 + 26}" '
            f'class="node-sublabel">{sublabel}</text>'
        )
    return "\n".join(parts)


def box(label, x, y, w, h, color, sublabel=None):
    """Generate a labelled rounded rectangle (software component box)."""
    parts = [
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" '
        f'fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="1.8"/>',
        f'  <text x="{x + w/2}" y="{y + h/2 + 4}" fill="{color}" '
        f'font-size="12" font-weight="600" text-anchor="middle">{label}</text>',
    ]
    if sublabel:
        parts.append(
            f'  <text x="{x + w/2}" y="{y + h/2 + 17}" fill="#777" '
            f'font-size="9" text-anchor="middle">{sublabel}</text>'
        )
    return "\n".join(parts)


def arrow(
    d, color_class, flow_class, marker_id, label=None, label_x=None, label_y=None
):
    """Generate an animated SVG arrow path."""
    parts = [
        f'  <path d="{d}" class="flow-arrow {flow_class} {color_class}" '
        f'marker-end="url(#{marker_id})"/>'
    ]
    if label and label_x and label_y:
        parts.append(
            f'  <text x="{label_x}" y="{label_y}" class="arrow-label">{label}</text>'
        )
    return "\n".join(parts)


html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MLServer — Software Architecture</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #f5f6f8; display: flex; justify-content: center;
         align-items: center; min-height: 100vh;
         font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }}
  svg {{ max-width: 1200px; width: 100%; height: auto; }}

  .flow-arrow {{
    fill: none; stroke-width: 2; stroke-dasharray: 8 6;
    stroke-linecap: round;
  }}
  .flow-right  {{ animation: dash-flow 1.2s linear infinite; }}
  .flow-left   {{ animation: dash-flow-rev 1.2s linear infinite; }}
  .flow-down   {{ animation: dash-flow 1.4s linear infinite; }}
  .flow-up     {{ animation: dash-flow-rev 1.4s linear infinite; }}
  .flow-slow   {{ animation: dash-flow 2.2s linear infinite; }}
  @keyframes dash-flow     {{ to {{ stroke-dashoffset: -28; }} }}
  @keyframes dash-flow-rev {{ to {{ stroke-dashoffset: 28; }} }}

  .arrow-blue   {{ stroke: #2E6EB5; }}
  .arrow-green  {{ stroke: #4B9A1E; }}
  .arrow-orange {{ stroke: #D4880F; }}
  .arrow-red    {{ stroke: #C0392B; }}
  .arrow-purple {{ stroke: #7D3C98; }}
  .arrow-teal   {{ stroke: #117A65; }}
  .arrow-cyan   {{ stroke: #0097A7; }}
  .arrow-gold   {{ stroke: #B58900; }}

  .cluster {{
    fill: none; stroke-width: 1.5; stroke-dasharray: 6 4;
    rx: 12; ry: 12; opacity: 0.7;
  }}
  .cluster-solid {{
    stroke-width: 1.5; rx: 14; ry: 14; opacity: 0.10;
  }}

  .cluster-title {{
    font-size: 13px; font-weight: 700;
    text-anchor: middle; opacity: 1;
  }}
  .cluster-note {{
    fill: #777; font-size: 9.5px; text-anchor: middle;
  }}
  .node-label {{
    font-size: 11px; font-weight: 600; text-anchor: middle;
  }}
  .node-sublabel {{
    fill: #777; font-size: 9.5px; text-anchor: middle;
  }}
  .arrow-label {{
    fill: #555; font-size: 9px; text-anchor: middle;
    font-style: italic;
  }}
</style>
</head>
<body>
<svg viewBox="0 0 1150 800" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="ah-blue" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2E6EB5"/></marker>
    <marker id="ah-green" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#4B9A1E"/></marker>
    <marker id="ah-orange" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#D4880F"/></marker>
    <marker id="ah-red" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#C0392B"/></marker>
    <marker id="ah-purple" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#7D3C98"/></marker>
    <marker id="ah-teal" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#117A65"/></marker>
    <marker id="ah-cyan" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#0097A7"/></marker>
    <marker id="ah-gold" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#B58900"/></marker>
  </defs>

  <!-- ═══ TITLE ═══ -->
  <text x="575" y="30" fill="#222" font-size="20" font-weight="700"
    text-anchor="middle">MLServer — Software Architecture</text>
  <text x="575" y="48" fill="#666" font-size="11"
    text-anchor="middle">V2 Inference Protocol  ·  REST + gRPC  ·  multi-frontend shared inference pool</text>

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  EXTERNAL ACTORS (left column)                        -->
  <!-- ═══════════════════════════════════════════════════════ -->

  {node("users", "Client", 75, 175, sublabel="Applications")}
  {node("prometheus", "Prometheus", 75, 295, sublabel="Scrape /metrics")}
  {node("storage", "Model Store", 75, 520, sublabel="PVC · S3 · Filesystem")}
  {node("cm", "Security", 75, 600, sublabel="Trusted Runtimes")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  FRONTEND PROCESSES (SO_REUSEPORT)                     -->
  <!-- ═══════════════════════════════════════════════════════ -->
  <rect x="175" y="115" width="330" height="315" class="cluster-solid"
    fill="#2E6EB5" stroke="#2E6EB5"/>
  <text x="340" y="137" class="cluster-title" fill="#2E6EB5">Frontend Processes × N</text>
  <text x="340" y="152" class="cluster-note">SO_REUSEPORT · same ports · frontend_workers = N</text>

  {box("REST API", 195, 165, 130, 50, "#2E6EB5", ":8080 · HTTP/JSON")}
  {box("gRPC API", 355, 165, 135, 50, "#2E6EB5", ":8081 · HTTP/2+Protobuf")}
  {box("DataPlane", 355, 230, 135, 50, "#4B9A1E", "Inference Orchestration")}
  {box("Metrics", 195, 295, 130, 50, "#D4880F", ":8082 · Prometheus")}
  {box("Middleware", 355, 295, 135, 50, "#4B9A1E", "CloudEvents · Hooks")}
  {box("Dispatcher", 355, 360, 135, 50, "#4B9A1E", "round-robin · model-affine")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  SUPERVISOR PROCESS                                    -->
  <!-- ═══════════════════════════════════════════════════════ -->
  <rect x="175" y="465" width="330" height="140" class="cluster-solid"
    fill="#B58900" stroke="#B58900"/>
  <text x="340" y="487" class="cluster-title" fill="#B58900">Supervisor Process</text>
  <text x="340" y="502" class="cluster-note">the main process at frontend_workers = 1</text>

  {box("FrontendSupervisor", 195, 512, 140, 55, "#B58900", "lifecycle · restarts")}
  {box("Shared Pool", 355, 512, 140, 55, "#7D3C98", "one copy of every model")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  INFERENCE WORKERS (shared)                            -->
  <!-- ═══════════════════════════════════════════════════════ -->
  <rect x="560" y="115" width="260" height="315" class="cluster-solid"
    fill="#C0392B" stroke="#C0392B"/>
  <text x="690" y="137" class="cluster-title" fill="#C0392B">Inference Workers (shared)</text>
  <text x="690" y="152" class="cluster-note">owned by the supervisor's pool</text>

  {box("Worker Processes", 580, 165, 220, 50, "#C0392B", "multiprocessing")}
  {box("Worker Registry", 580, 230, 220, 50, "#7D3C98", "batching hooks installed")}
  {box("Adaptive Batcher", 580, 295, 220, 50, "#7D3C98", "lives next to the model")}
  {box("Codec Pipeline", 580, 360, 220, 50, "#117A65", "ndarray-native")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  EXTERNAL SYSTEMS (right column)                       -->
  <!-- ═══════════════════════════════════════════════════════ -->

  {node("deploy", "KServe", 975, 640, sublabel="Controller · CRD")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  SHIPPED RUNTIMES                                      -->
  <!-- ═══════════════════════════════════════════════════════ -->
  <rect x="175" y="650" width="645" height="95" class="cluster-solid"
    fill="#0097A7" stroke="#0097A7"/>
  <text x="497" y="670" class="cluster-title" fill="#0097A7">Shipped Runtimes (Production)</text>

  {node("pod", "sklearn", 280, 710, size=44, sublabel="SKLearnModel")}
  {node("pod", "xgboost", 420, 710, size=44, sublabel="XGBoostModel")}
  {node("pod", "lightgbm", 560, 710, size=44, sublabel="LightGBMModel")}
  {node("pod", "onnx", 700, 710, size=44, sublabel="OnnxModel")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  ANIMATED FLOW ARROWS                                  -->
  <!-- ═══════════════════════════════════════════════════════ -->

  <!-- Client → REST -->
  {arrow("M 101,180 L 191,180", "arrow-blue", "flow-right", "ah-blue")}
  <!-- Client → gRPC (over the top, into the cluster) -->
  {arrow("M 90,148 L 90,102 L 422,102 L 422,161", "arrow-blue", "flow-right", "ah-blue")}
  <!-- Response: REST → Client -->
  {arrow("M 193,212 L 101,212", "arrow-green", "flow-left", "ah-green")}

  <!-- Prometheus → Metrics -->
  {arrow("M 101,310 L 191,310", "arrow-red", "flow-left", "ah-red",
         "scrape", 146, 300)}

  <!-- REST → DataPlane -->
  {arrow("M 327,190 L 351,190", "arrow-green", "flow-right", "ah-green")}
  <!-- gRPC → DataPlane -->
  {arrow("M 422,217 L 422,226", "arrow-green", "flow-down", "ah-green")}
  <!-- DataPlane → Middleware -->
  {arrow("M 422,282 L 422,291", "arrow-green", "flow-down", "ah-green")}
  <!-- Middleware → Dispatcher -->
  {arrow("M 422,347 L 422,356", "arrow-green", "flow-down", "ah-green")}

  <!-- Dispatcher → Workers (ZeroMQ data plane) -->
  {arrow("M 492,385 L 530,385 L 530,190 L 576,190", "arrow-red", "flow-right", "ah-red",
         "ZeroMQ", 530, 170)}
  <!-- Workers → Dispatcher (replies routed per frontend) -->
  {arrow("M 576,215 L 545,215 L 545,400 L 492,400", "arrow-red", "flow-slow", "ah-red",
         "replies", 548, 300)}

  <!-- Workers → Registry → Batcher → Codec -->
  {arrow("M 690,217 L 690,226", "arrow-purple", "flow-down", "ah-purple")}
  {arrow("M 690,282 L 690,291", "arrow-purple", "flow-down", "ah-purple")}
  {arrow("M 690,347 L 690,356", "arrow-purple", "flow-down", "ah-purple")}

  <!-- Supervisor → Frontends (control pipes) -->
  {arrow("M 270,510 L 270,434", "arrow-gold", "flow-up", "ah-gold",
         "model lifecycle (pipes)", 390, 450)}
  <!-- FrontendSupervisor → Shared Pool -->
  {arrow("M 337,539 L 351,539", "arrow-gold", "flow-right", "ah-gold")}
  <!-- Shared Pool → Workers (owns) -->
  {arrow("M 497,539 L 660,539 L 660,434", "arrow-purple", "flow-up", "ah-purple",
         "owns", 660, 505)}

  <!-- Supervisor → Model Store -->
  {arrow("M 193,527 L 101,527", "arrow-orange", "flow-left", "ah-orange",
         "load artifacts", 147, 517)}

  <!-- KServe → Supervisor -->
  {arrow("M 949,640 L 340,640 L 340,609", "arrow-blue", "flow-left", "ah-blue",
         "manage models", 640, 630)}

  <!-- Security → Supervisor (runtime allowlist) -->
  {arrow("M 99,590 L 171,590", "arrow-red", "flow-right", "ah-red",
         "allowlist", 135, 606)}

  <!-- Workers → Runtimes (fan-out) -->
  {arrow("M 600,432 L 600,622 L 280,622 L 280,684", "arrow-cyan", "flow-down", "ah-cyan")}
  {arrow("M 625,432 L 625,622 L 420,622 L 420,684", "arrow-cyan", "flow-down", "ah-cyan")}
  {arrow("M 650,432 L 650,622 L 560,622 L 560,684", "arrow-cyan", "flow-down", "ah-cyan")}
  {arrow("M 675,432 L 675,622 L 700,622 L 700,684", "arrow-cyan", "flow-down", "ah-cyan")}

  <!-- ═══ FOOTER ═══ -->
  <text x="575" y="775" fill="#888" font-size="10" text-anchor="middle">
    opendatahub-io/MLServer  ·  V2 Inference Protocol  ·  Apache-2.0
  </text>
  <text x="575" y="788" fill="#999" font-size="9" text-anchor="middle">
    Shipped runtimes: sklearn · xgboost · lightgbm · onnx
  </text>

</svg>
</body>
</html>"""

with open(OUT_FILE, "w") as f:
    f.write(html)

print(f"Written to {OUT_FILE}")
print(f"File size: {os.path.getsize(OUT_FILE) / 1024:.0f} KB")
