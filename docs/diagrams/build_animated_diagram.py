"""
Build the animated MLServer architecture diagram as a self-contained HTML file.

Uses Kubernetes-style node icons from the diagrams library (Apache-2.0 licensed)
and Simple Icons brand logos (MIT licensed). All icons are embedded as base64
data URIs so the HTML file is fully self-contained.

Usage:
    python3 build_animated_diagram.py
"""

import base64
import os

ICONS_DIR = os.path.join(os.path.dirname(__file__), "icons")
OUT_FILE = os.path.join(os.path.dirname(__file__), "deployment_topology_animated.html")


def b64_img(filename):
    """Read a PNG file and return a base64 data URI."""
    path = os.path.join(ICONS_DIR, filename)
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# Load all icons as data URIs
icons = {
    "pod": b64_img("pod.png"),
    "svc": b64_img("svc.png"),
    "ing": b64_img("ing.png"),
    "pvc": b64_img("pvc.png"),
    "cm": b64_img("cm.png"),
    "deploy": b64_img("deploy.png"),
    "prometheus": b64_img("prometheus.png"),
    "grafana": b64_img("grafana.png"),
    "kafka": b64_img("kafka.png"),
    "users": b64_img("users.png"),
    "storage": b64_img("storage.png"),
}


def node(icon_key, label, x, y, size=52, sublabel=None, label_color="#333"):
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
            f'  <text x="{x}" y="{y + size/2 + 27}" '
            f'class="node-sublabel">{sublabel}</text>'
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
<title>MLServer — Production Deployment Topology</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #f5f6f8; display: flex; justify-content: center;
         align-items: center; min-height: 100vh;
         font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }}
  svg {{ max-width: 1150px; width: 100%; height: auto; }}

  .flow-arrow {{
    fill: none; stroke-width: 2; stroke-dasharray: 8 6;
    stroke-linecap: round;
  }}
  .flow-right  {{ animation: dash-flow 1.2s linear infinite; }}
  .flow-left   {{ animation: dash-flow-rev 1.2s linear infinite; }}
  .flow-down   {{ animation: dash-flow 1.4s linear infinite; }}
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
<svg viewBox="0 0 1100 820" xmlns="http://www.w3.org/2000/svg">
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
  <text x="550" y="32" fill="#222" font-size="20" font-weight="700"
    text-anchor="middle">MLServer — Production Deployment Topology</text>
  <text x="550" y="50" fill="#666" font-size="11"
    text-anchor="middle">V2 Inference Protocol  ·  REST + gRPC + Kafka  ·  Kubernetes Native</text>

  <!-- ═══ INFERENCE CLIENTS (external) ═══ -->
{node("users", "Inference Clients", 65, 100)}

  <!-- ═══ INGRESS (external) ═══ -->
{node("ing", "Ingress / Route", 65, 225, sublabel="ing")}

  <!-- ═══ KUBERNETES CLUSTER ═══ -->
  <rect x="130" y="68" width="950" height="700" class="cluster-solid"
    fill="#4A90D9" stroke="#2E6EB5"/>
  <text x="605" y="86" class="cluster-title" fill="#2E6EB5">Kubernetes Cluster</text>

  <!-- ═══ KServe Control Plane ═══ -->
  <rect x="530" y="95" width="140" height="115" class="cluster" stroke="#2E6EB5"/>
  <text x="600" y="112" class="cluster-title" fill="#2E6EB5">KServe Control Plane</text>
{node("deploy", "KServe Controller", 600, 165, sublabel="deploy")}

  <!-- ═══ Observability ═══ -->
  <rect x="700" y="95" width="170" height="115" class="cluster" stroke="#C0392B"/>
  <text x="785" y="112" class="cluster-title" fill="#C0392B">Observability</text>
{node("prometheus", "Prometheus", 750, 165)}
{node("grafana", "Grafana", 830, 165, size=46)}

  <!-- ═══ Messaging ═══ -->
  <rect x="895" y="95" width="115" height="115" class="cluster" stroke="#7D3C98"/>
  <text x="952" y="112" class="cluster-title" fill="#7D3C98">Messaging (optional)</text>
{node("kafka", "Kafka Broker", 952, 165)}

  <!-- ═══ Security ═══ -->
  <rect x="895" y="225" width="165" height="95" class="cluster" stroke="#C0392B"/>
  <text x="977" y="242" class="cluster-title" fill="#C0392B">Security</text>
{node("cm", "trusted-runtimes.json", 977, 282, sublabel="(/etc/mlserver/)")}

  <!-- ═══ MLServer Pod ═══ -->
  <rect x="150" y="220" width="530" height="105" class="cluster-solid"
    fill="#2E6EB5" stroke="#2E6EB5"/>
  <text x="415" y="238" class="cluster-title" fill="#2E6EB5">MLServer Pod</text>

{node("svc", "REST :8080", 210, 285, sublabel="SVC")}
{node("svc", "gRPC :8081", 330, 285, sublabel="SVC")}
{node("svc", "Metrics :8082", 450, 285, sublabel="SVC")}
{node("cm", "InferenceService", 580, 285, sublabel="CRD")}

  <!-- ═══ MLServer Container ═══ -->
  <rect x="150" y="345" width="530" height="145" class="cluster-solid"
    fill="#4B9A1E" stroke="#4B9A1E"/>
  <text x="415" y="363" class="cluster-title" fill="#4B9A1E">MLServer Container</text>

{node("pod", "MLServer", 415, 425, size=56, sublabel="(main process)")}

  <!-- ═══ Shipped Runtimes ═══ -->
  <rect x="150" y="510" width="530" height="120" class="cluster-solid"
    fill="#0097A7" stroke="#0097A7"/>
  <text x="415" y="528" class="cluster-title" fill="#0097A7">Shipped Runtimes</text>

{node("pod", "sklearn", 225, 580)}
{node("pod", "xgboost", 345, 580)}
{node("pod", "lightgbm", 465, 580)}
{node("pod", "onnx", 585, 580)}

  <!-- ═══ Storage ═══ -->
  <rect x="700" y="510" width="315" height="120" class="cluster" stroke="#D4880F"/>
  <text x="857" y="528" class="cluster-title" fill="#D4880F">Storage</text>

{node("pvc", "Model Store", 780, 580, sublabel="(PVC)")}
{node("storage", "S3 / MinIO", 910, 580, sublabel="(remote)")}

  <!-- ═══════════════════════════════════════════════════════ -->
  <!--  ANIMATED FLOW ARROWS                                  -->
  <!-- ═══════════════════════════════════════════════════════ -->

  <!-- Client → Ingress (vertical) -->
{arrow("M 65,135 L 65,190", "arrow-blue", "flow-down", "ah-blue",
       "HTTP / gRPC", 45, 165)}

  <!-- Ingress → REST SVC (right then down) -->
{arrow("M 95,225 L 210,225 L 210,258", "arrow-blue", "flow-right", "ah-blue")}

  <!-- Ingress → gRPC SVC (right then down) -->
{arrow("M 95,235 L 330,235 L 330,258", "arrow-blue", "flow-right", "ah-blue")}

  <!-- REST SVC → MLServer pod (down then right) -->
{arrow("M 210,315 L 210,400 L 380,400", "arrow-green", "flow-down", "ah-green")}

  <!-- gRPC SVC → MLServer pod (down then right) -->
{arrow("M 330,315 L 330,400 L 390,400", "arrow-green", "flow-down", "ah-green")}

  <!-- Metrics SVC → MLServer pod (down) -->
{arrow("M 450,315 L 450,400 L 440,400", "arrow-green", "flow-down", "ah-green")}

  <!-- KServe → InferenceService CRD (down then left) -->
{arrow("M 600,200 L 600,258", "arrow-blue", "flow-down", "ah-blue",
       "reconcile", 618, 230)}

  <!-- InferenceService CRD → MLServer pod (down then left) -->
{arrow("M 580,315 L 580,400 L 445,400", "arrow-blue", "flow-down", "ah-blue",
       "manage pod", 530, 360)}

  <!-- Prometheus → Metrics SVC (left then down) -->
{arrow("M 720,165 L 480,165 L 480,258", "arrow-red", "flow-left", "ah-red",
       "scrape :8082", 600, 155)}

  <!-- Prometheus → Grafana (horizontal) -->
{arrow("M 780,165 L 805,165", "arrow-red", "flow-right", "ah-red")}

  <!-- Kafka → MLServer (down then left) -->
{arrow("M 952,200 L 952,425 L 445,425", "arrow-purple", "flow-slow", "ah-purple",
       "consume / produce", 700, 415)}

  <!-- Security ConfigMap → MLServer (left then down) -->
{arrow("M 895,282 L 680,282 L 680,425 L 448,425", "arrow-red", "flow-left", "ah-red",
       "mount", 790, 272)}

  <!-- MLServer → Runtimes (down then left/right) -->
{arrow("M 380,460 L 380,490 L 225,490 L 225,548", "arrow-cyan", "flow-down", "ah-cyan")}
{arrow("M 395,460 L 395,490 L 345,490 L 345,548", "arrow-cyan", "flow-down", "ah-cyan")}
{arrow("M 435,460 L 435,490 L 465,490 L 465,548", "arrow-cyan", "flow-down", "ah-cyan")}
{arrow("M 450,460 L 450,490 L 585,490 L 585,548", "arrow-cyan", "flow-down", "ah-cyan")}

  <!-- MLServer → Model Store (right then down) -->
{arrow("M 445,445 L 690,445 L 690,570 L 750,570", "arrow-orange", "flow-right", "ah-orange",
       "load artifacts", 690, 510)}

  <!-- MLServer → S3 (right then down) -->
{arrow("M 450,440 L 700,440 L 700,475 L 910,475 L 910,548", "arrow-orange", "flow-right", "ah-orange",
       "fetch remote", 810, 465)}

  <!-- Response path back: MLServer → REST → Ingress → Client -->
{arrow("M 395,395 L 195,395 L 195,310", "arrow-green", "flow-left", "ah-green")}
{arrow("M 195,270 L 80,270 L 80,240", "arrow-green", "flow-left", "ah-green")}
{arrow("M 65,200 L 65,135", "arrow-green", "flow-up", "ah-green")}

  <!-- ═══ FOOTER ═══ -->
  <text x="550" y="690" fill="#888" font-size="10" text-anchor="middle">
    opendatahub-io/MLServer  ·  V2 Inference Protocol  ·  Apache-2.0
  </text>
  <text x="550" y="705" fill="#999" font-size="9" text-anchor="middle">
    Shipped: sklearn · xgboost · lightgbm · onnx    |
    Community: catboost · mlflow · huggingface · alibi-detect · alibi-explain · mllib
  </text>
  <text x="550" y="720" fill="#aaa" font-size="8" text-anchor="middle">
    K8s icons: diagrams library (Apache-2.0)  ·  Brand logos: Simple Icons (MIT)
  </text>

</svg>
</body>
</html>"""

with open(OUT_FILE, "w") as f:
    f.write(html)

print(f"Written to {OUT_FILE}")
print(f"File size: {os.path.getsize(OUT_FILE) / 1024:.0f} KB")
