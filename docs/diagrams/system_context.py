"""
MLServer Software Architecture (C4-style System Context).

Generates a static PNG that is content-identical to the animated GIF
(architecture_animated.gif): same nodes, arrows, naming, and groupings.
The GIF is referenced in architecture.md; this PNG is kept as an asset
for use in contexts that cannot render GIFs (PDFs, slide decks, etc.).

Usage:
    python3 system_context.py
"""

import os

from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.onprem.monitoring import Prometheus
from diagrams.programming.framework import FastAPI
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL
from diagrams.k8s.compute import Pod, Deployment
from diagrams.k8s.podconfig import ConfigMap

graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.8",
    "nodesep": "0.9",
    "ranksep": "0.9",
    "splines": "ortho",
}

with Diagram(
    "MLServer — Software Architecture",
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_context"),
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    outformat="png",
):
    # ── External actors (left) ─────────────────────────────────
    client = Users("Client\nApplications")
    prometheus = Prometheus("Prometheus\nScrape /metrics")
    security = ConfigMap("Security\nTrusted Runtimes")

    # ── External actors (right) ────────────────────────────────
    model_store = Storage("Model Store\nPVC · S3 · Filesystem")
    kserve = Deployment("KServe\nController · CRD")

    with Cluster(
        "MLServer Boundary",
        graph_attr={"bgcolor": "#E8F4FD", "pencolor": "#4A90D9"},
    ):

        with Cluster(
            "Supervisor Process\n(at frontend_workers = 0 this is the main process)",
            graph_attr={"bgcolor": "#FDF6E3", "pencolor": "#B58900"},
        ):
            supervisor = Rack("FrontendSupervisor\nlifecycle coordination")
            pool = SQL("Shared Inference Pool\none copy of every model")

        with Cluster(
            "Frontend Processes × N\n(REST: shared listener)",
            graph_attr={"bgcolor": "#E8F4FD", "pencolor": "#4A90D9"},
        ):
            rest = FastAPI("REST API\n:8080 · HTTP/JSON")
            grpc = Server("gRPC API\n:8081 · HTTP/2+Protobuf")
            metrics = FastAPI("Metrics\n:8082 · Prometheus")
            dataplane = Rack("DataPlane\nInference Orchestration")
            middleware = Rack("Middleware\nCloudEvents · Hooks")
            dispatcher = Rack("Dispatcher\nZeroMQ replies routed\nper frontend")

        with Cluster(
            "Inference Workers (shared)",
            graph_attr={"bgcolor": "#F0FFF0", "pencolor": "#7ED321"},
        ):
            batcher = Rack(
                "Adaptive Batcher\nlives next to the model\n(model-affine routing)"
            )
            codec = Rack("Codec Pipeline\nndarray-native encode · decode")
            workers = Server("Worker Processes\nmultiprocessing")

        with Cluster("Shipped Runtimes (Production)"):
            sklearn = Pod("sklearn\nSKLearnModel")
            xgboost = Pod("xgboost\nXGBoostModel")
            lightgbm = Pod("lightgbm\nLightGBMModel")
            onnx = Pod("onnx\nOnnxModel")

    # ── Client → Frontends (kernel-level connection balancing) ─
    client >> Edge(label="HTTP/JSON", color="#2E6EB5", style="bold") >> rest
    client >> Edge(label="HTTP/2+Protobuf", color="#2E6EB5", style="bold") >> grpc

    # ── Prometheus → Metrics ───────────────────────────────────
    (
        prometheus
        >> Edge(label="scrape :8082", color="#C0392B", style="dashed")
        >> metrics
    )

    # ── Security → Supervisor ─────────────────────────────────
    (security >> Edge(label="allowlist", color="#C0392B", style="dashed") >> supervisor)

    # ── Frontend request path ─────────────────────────────────
    rest >> Edge(color="#4B9A1E") >> dataplane
    grpc >> Edge(color="#4B9A1E") >> dataplane
    dataplane >> Edge(color="#4B9A1E") >> middleware
    middleware >> Edge(label="dispatch", color="#C0392B") >> dispatcher

    # ── Data plane: dispatcher → shared workers (ZeroMQ) ──────
    dispatcher >> Edge(label="ZeroMQ data plane", color="#C0392B") >> workers

    # ── In-worker execution ────────────────────────────────────
    workers >> Edge(color="#7D3C98") >> batcher
    batcher >> Edge(color="#7D3C98") >> codec

    # ── Control plane: supervisor ↔ frontends (pipes) ─────────
    (
        supervisor
        >> Edge(label="model lifecycle\n(apply · ack, pipes)", color="#B58900")
        >> dataplane
    )
    supervisor >> Edge(color="#B58900") >> pool

    # ── KServe → Supervisor ────────────────────────────────────
    (
        kserve
        >> Edge(label="manage models", color="#2E6EB5", style="dashed")
        >> supervisor
    )

    # ── Supervisor → Model Store ───────────────────────────────
    (
        supervisor
        >> Edge(label="load artifacts", color="#D4880F", style="dashed")
        >> model_store
    )

    # ── Workers → Runtimes (fan-out) ───────────────────────────
    workers >> Edge(color="#0097A7") >> sklearn
    workers >> Edge(color="#0097A7") >> xgboost
    workers >> Edge(color="#0097A7") >> lightgbm
    workers >> Edge(color="#0097A7") >> onnx

    # ── Response path ──────────────────────────────────────────
    rest >> Edge(color="#4B9A1E", style="dashed") >> client
