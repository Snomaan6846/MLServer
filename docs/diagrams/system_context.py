"""
MLServer Software Architecture (C4-style System Context).

Generates a static PNG that is content-identical to the animated GIF
(architecture_animated.gif): same nodes, arrows, naming, and groupings.
The GIF is referenced in architecture.md; this PNG is kept as an asset
for use in contexts that cannot render GIFs (PDFs, slide decks, etc.).

Usage:
    python3 system_context.py
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.onprem.monitoring import Prometheus
from diagrams.onprem.queue import Kafka
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
    filename="/Users/imran/Projects/MLServer/docs/diagrams/system_context",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    outformat="png",
):
    # ── External actors (left) ─────────────────────────────────
    client = Users("Client\nApplications")
    kafka_ext = Kafka("Kafka\nMessage Bus")
    prometheus = Prometheus("Prometheus\nScrape /metrics")
    security = ConfigMap("Security\nTrusted Runtimes")

    # ── External actors (right) ────────────────────────────────
    model_store = Storage("Model Store\nPVC · S3 · Filesystem")
    kserve = Deployment("KServe\nController · CRD")

    with Cluster(
        "MLServer Boundary",
        graph_attr={"bgcolor": "#E8F4FD", "pencolor": "#4A90D9"},
    ):

        with Cluster("Transport Layer"):
            rest = FastAPI("REST API\n:8080 · HTTP/JSON")
            grpc = Server("gRPC API\n:8081 · HTTP/2+Protobuf")
            metrics = FastAPI("Metrics\n:8082 · Prometheus")
            kafka_srv = Server("Kafka Consumer\nCloudEvents")

        with Cluster("Core Engine"):
            dataplane = Rack("DataPlane\nInference Orchestration")
            middleware = Rack("Middleware\nCloudEvents · Hooks")
            cache = SQL("Response Cache\nLRU · Per-request key")

        with Cluster("Model Management"):
            registry = SQL("Registry\nMultiModel · Versioned")
            batcher = Rack("Batcher\nAdaptive Batching")
            codec = Rack("Codec Pipeline\nEncode · Decode")

        with Cluster("Inference Execution"):
            workers = Server("Workers\nMultiprocessing")
            queues = Rack("Queues\nReq/Resp · Dispatch")

        with Cluster("Shipped Runtimes (Production)"):
            sklearn = Pod("sklearn\nSKLearnModel")
            xgboost = Pod("xgboost\nXGBoostModel")
            lightgbm = Pod("lightgbm\nLightGBMModel")
            onnx = Pod("onnx\nOnnxModel")

        with Cluster(
            "Community Runtimes (Source Only — Not Shipped)",
            graph_attr={"style": "dashed", "pencolor": "#888888"},
        ):
            catboost = Pod("catboost")
            mlflow = Pod("mlflow")
            huggingface = Pod("huggingface")
            alibi_detect = Pod("alibi-detect")
            alibi_explain = Pod("alibi-explain")
            mllib = Pod("mllib")

    # ── Client → Transport ─────────────────────────────────────
    client >> Edge(label="HTTP/JSON", color="#2E6EB5", style="bold") >> rest
    client >> Edge(label="HTTP/2+Protobuf", color="#2E6EB5", style="bold") >> grpc

    # ── Kafka external → Kafka Consumer ────────────────────────
    (
        kafka_ext
        >> Edge(label="consume / produce", color="#7D3C98", style="dashed")
        >> kafka_srv
    )

    # ── Prometheus → Metrics ───────────────────────────────────
    (
        prometheus
        >> Edge(label="scrape :8082", color="#C0392B", style="dashed")
        >> metrics
    )

    # ── Security → Middleware ──────────────────────────────────
    (security >> Edge(label="allowlist", color="#C0392B", style="dashed") >> middleware)

    # ── Transport → Core Engine ────────────────────────────────
    rest >> Edge(color="#4B9A1E") >> dataplane
    grpc >> Edge(color="#4B9A1E") >> dataplane
    kafka_srv >> Edge(label="CloudEvents", color="#7D3C98") >> middleware

    # ── Core Engine internal ───────────────────────────────────
    dataplane >> Edge(color="#4B9A1E") >> middleware
    dataplane >> Edge(label="cache lookup", color="#117A65") >> cache

    # ── Core Engine → Model Management ─────────────────────────
    dataplane >> Edge(label="resolve model", color="#7D3C98") >> registry
    registry >> Edge(color="#7D3C98") >> batcher
    batcher >> Edge(color="#7D3C98") >> codec

    # ── Model Management → Execution ──────────────────────────
    codec >> Edge(label="dispatch", color="#C0392B") >> workers
    workers >> Edge(color="#C0392B") >> queues

    # ── KServe → Registry ─────────────────────────────────────
    (kserve >> Edge(label="manage models", color="#2E6EB5", style="dashed") >> registry)

    # ── Registry → Model Store ─────────────────────────────────
    (
        registry
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
