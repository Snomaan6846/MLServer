"""
Build the animated MLServer Software Architecture diagram as a self-contained HTML file.

Shows the internal component perspective: DataPlane, Middleware, Response Cache,
Model Registry, Workers, Queues, and runtime plugins. Complements the deployment
topology diagram which shows the K8s infrastructure perspective.

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
    "kafka": b64_img("kafka.png"),
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
<svg viewBox="0 0 1150 780" xmlns="http://www.w3.org/2000/svg">
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
  </defs>

  <!-- ═══ TITLE ═══ -->
  <text x="575" y="32" fill="#222" font-size="20" font-weight="700"
    text-anchor="middle">MLServer — Software Architecture</text>
  <text x="575" y="50" fill="#666" font-size="11"
    text-anchor="middle">V2 Inference Protocol  ·  REST + gRPC + Kafka</text>

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  EXTERNAL CLIENTS (left column)                        -->
  <!-- ═══════════════════════════════════════════════════════ -->

{node("users", "Client", 80, 200, sublabel="Applications")}
{node("kafka", "Kafka", 80, 380, sublabel="Message Bus")}
{node("prometheus", "Prometheus", 80, 520, sublabel="Scrape /metrics")}
{node("cm", "Security", 80, 650, sublabel="Trusted Runtimes")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  TRANSPORT LAYER                                       -->
  <!-- ═══════════════════════════════════════════════════════ -->
  <rect x="175" y="120" width="200" height="300" class="cluster-solid"
    fill="#2E6EB5" stroke="#2E6EB5"/>
  <text x="275" y="140" class="cluster-title" fill="#2E6EB5">Transport Layer</text>

{box("REST API", 195, 160, 160, 50, "#2E6EB5", ":8080 · HTTP/JSON")}
{box("gRPC API", 195, 230, 160, 50, "#2E6EB5", ":8081 · HTTP/2+Protobuf")}
{box("Metrics", 195, 300, 160, 50, "#D4880F", ":8082 · Prometheus")}
{box("Kafka Consumer", 195, 370, 160, 50, "#7D3C98", "CloudEvents")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  CORE ENGINE                                           -->
  <!-- ═══════════════════════════════════════════════════════ -->
  <rect x="405" y="120" width="200" height="300" class="cluster-solid"
    fill="#4B9A1E" stroke="#4B9A1E"/>
  <text x="505" y="140" class="cluster-title" fill="#4B9A1E">Core Engine</text>

{box("DataPlane", 425, 160, 160, 50, "#4B9A1E", "Inference Orchestration")}
{box("Middleware", 425, 230, 160, 50, "#4B9A1E", "CloudEvents · Hooks")}
{box("Response Cache", 425, 300, 160, 50, "#117A65", "LRU · Per-request key")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  MODEL MANAGEMENT                                      -->
  <!-- ═══════════════════════════════════════════════════════ -->
  <rect x="635" y="120" width="200" height="230" class="cluster-solid"
    fill="#7D3C98" stroke="#7D3C98"/>
  <text x="735" y="140" class="cluster-title" fill="#7D3C98">Model Management</text>

{box("Registry", 655, 160, 160, 50, "#7D3C98", "MultiModel · Versioned")}
{box("Batcher", 655, 230, 160, 50, "#7D3C98", "Adaptive Batching")}
{box("Codec Pipeline", 655, 300, 160, 50, "#117A65", "Encode · Decode")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  EXTERNAL SYSTEMS (right column)                       -->
  <!-- ═══════════════════════════════════════════════════════ -->

{node("storage", "Model Store", 960, 200, sublabel="PVC · S3 · Filesystem")}
{node("deploy", "KServe", 960, 350, sublabel="Controller · CRD")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  INFERENCE EXECUTION                                   -->
  <!-- ═══════════════════════════════════════════════════════ -->
  <rect x="635" y="375" width="280" height="85" class="cluster-solid"
    fill="#C0392B" stroke="#C0392B"/>
  <text x="775" y="392" class="cluster-title" fill="#C0392B">Inference Execution</text>

{box("Workers", 655, 405, 115, 45, "#C0392B", "Multiprocessing")}
{box("Queues", 785, 405, 115, 45, "#C0392B", "Req/Resp Dispatch")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  SHIPPED RUNTIMES                                      -->
  <!-- ═══════════════════════════════════════════════════════ -->
  <rect x="200" y="500" width="640" height="105" class="cluster-solid"
    fill="#0097A7" stroke="#0097A7"/>
  <text x="520" y="518" class="cluster-title" fill="#0097A7">Shipped Runtimes (Production)</text>

{node("pod", "sklearn", 280, 565, size=44, sublabel="SKLearnModel")}
{node("pod", "xgboost", 415, 565, size=44, sublabel="XGBoostModel")}
{node("pod", "lightgbm", 550, 565, size=44, sublabel="LightGBMModel")}
{node("pod", "onnx", 685, 565, size=44, sublabel="OnnxModel")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  COMMUNITY RUNTIMES                                    -->
  <!-- ═══════════════════════════════════════════════════════ -->
  <rect x="200" y="625" width="780" height="95" class="cluster"
    stroke="#888"/>
  <text x="590" y="643" class="cluster-title" fill="#888">Community Runtimes (Source Only — Not Shipped)</text>

{node("pod", "catboost", 265, 690, size=38, label_color="#888")}
{node("pod", "mlflow", 380, 690, size=38, label_color="#888")}
{node("pod", "huggingface", 510, 690, size=38, label_color="#888")}
{node("pod", "alibi-detect", 650, 690, size=38, label_color="#888")}
{node("pod", "alibi-explain", 790, 690, size=38, label_color="#888")}
{node("pod", "mllib", 910, 690, size=38, label_color="#888")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  ANIMATED FLOW ARROWS                                  -->
  <!-- ═══════════════════════════════════════════════════════ -->

  <!-- Client → REST (right then up) -->
{arrow("M 112,195 L 155,195 L 155,180 L 193,180", "arrow-blue", "flow-right", "ah-blue")}
  <!-- Client → gRPC (right then down) -->
{arrow("M 112,210 L 155,210 L 155,250 L 193,250", "arrow-blue", "flow-right", "ah-blue")}

  <!-- REST → DataPlane (horizontal) -->
{arrow("M 357,185 L 423,185", "arrow-green", "flow-right", "ah-green")}
  <!-- gRPC → DataPlane (right then up) -->
{arrow("M 357,255 L 390,255 L 390,190 L 423,190", "arrow-green", "flow-right", "ah-green")}

  <!-- Kafka → Kafka Consumer (horizontal) -->
{arrow("M 112,380 L 193,380", "arrow-purple", "flow-right", "ah-purple")}
  <!-- Kafka Consumer → Middleware (right then up) -->
{arrow("M 357,395 L 390,395 L 390,255 L 423,255", "arrow-purple", "flow-right", "ah-purple",
       "CloudEvents", 390, 320)}

  <!-- Prometheus → Metrics (right then up) -->
{arrow("M 112,520 L 155,520 L 155,330 L 193,330", "arrow-red", "flow-left", "ah-red",
       "scrape", 155, 430)}

  <!-- Security → Middleware (right then up) -->
{arrow("M 112,650 L 155,650 L 155,268 L 423,268", "arrow-red", "flow-right", "ah-red",
       "allowlist", 155, 460)}

  <!-- DataPlane → Registry (horizontal) -->
{arrow("M 587,185 L 653,185", "arrow-purple", "flow-right", "ah-purple",
       "resolve model", 620, 175)}

  <!-- DataPlane → Response Cache (vertical) -->
{arrow("M 505,212 L 505,298", "arrow-teal", "flow-down", "ah-teal",
       "cache lookup", 545, 260)}

  <!-- DataPlane → Middleware (vertical) -->
{arrow("M 505,212 L 505,228", "arrow-green", "flow-down", "ah-green")}

  <!-- Registry → Model Store (horizontal) -->
{arrow("M 817,185 L 928,185", "arrow-orange", "flow-right", "ah-orange",
       "load artifacts", 875, 175)}

  <!-- Registry → Batcher (vertical) -->
{arrow("M 735,212 L 735,228", "arrow-purple", "flow-down", "ah-purple")}

  <!-- Batcher → Codec Pipeline (vertical) -->
{arrow("M 735,282 L 735,298", "arrow-purple", "flow-down", "ah-purple")}

  <!-- Codec Pipeline → Workers (down then left) -->
{arrow("M 735,352 L 735,405 L 772,405", "arrow-red", "flow-down", "ah-red",
       "dispatch", 750, 380)}

  <!-- Workers → Queues (horizontal) -->
{arrow("M 772,427 L 783,427", "arrow-red", "flow-right", "ah-red")}

  <!-- KServe → Registry (left then up) -->
{arrow("M 928,345 L 860,345 L 860,200 L 817,200", "arrow-blue", "flow-left", "ah-blue",
       "manage models", 860, 270)}

  <!-- Workers → Runtimes (down then left/right) -->
{arrow("M 680,452 L 680,490 L 280,490 L 280,538", "arrow-cyan", "flow-down", "ah-cyan")}
{arrow("M 695,452 L 695,490 L 415,490 L 415,538", "arrow-cyan", "flow-down", "ah-cyan")}
{arrow("M 720,452 L 720,490 L 550,490 L 550,538", "arrow-cyan", "flow-down", "ah-cyan")}
{arrow("M 740,452 L 740,490 L 685,490 L 685,538", "arrow-cyan", "flow-down", "ah-cyan")}

  <!-- Response path: DataPlane → REST → Client -->
{arrow("M 423,195 L 357,195", "arrow-green", "flow-left", "ah-green")}
{arrow("M 193,195 L 112,195", "arrow-green", "flow-left", "ah-green")}

  <!-- ═══ FOOTER ═══ -->
  <text x="575" y="750" fill="#888" font-size="10" text-anchor="middle">
    opendatahub-io/MLServer  ·  V2 Inference Protocol  ·  Apache-2.0
  </text>
  <text x="575" y="765" fill="#999" font-size="9" text-anchor="middle">
    Shipped: sklearn · xgboost · lightgbm · onnx    |
    Community: catboost · mlflow · huggingface · alibi-detect · alibi-explain · mllib
  </text>

</svg>
</body>
</html>"""

with open(OUT_FILE, "w") as f:
    f.write(html)

print(f"Written to {OUT_FILE}")
print(f"File size: {os.path.getsize(OUT_FILE) / 1024:.0f} KB")
