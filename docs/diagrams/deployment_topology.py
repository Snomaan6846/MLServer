"""MLServer Production Deployment Topology on Kubernetes.

Shows the pod layout at frontend_workers > 1: a supervisor process
owning the shared inference pool, N frontend processes accepting
REST/gRPC/metrics traffic on the same ports (SO_REUSEPORT), and the
shared worker processes. At frontend_workers = 1 all three collapse
into a single main process.
"""

import os

from diagrams import Cluster, Diagram, Edge
from diagrams.k8s.compute import Pod, Deployment
from diagrams.k8s.network import Service, Ingress
from diagrams.k8s.storage import PersistentVolumeClaim as PVC
from diagrams.k8s.podconfig import ConfigMap
from diagrams.onprem.monitoring import Prometheus, Grafana
from diagrams.onprem.client import Users
from diagrams.generic.storage import Storage

graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.8",
    "nodesep": "0.8",
    "ranksep": "1.2",
}

with Diagram(
    "MLServer — Production Deployment Topology",
    filename=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "deployment_topology"
    ),
    show=False,
    direction="TB",
    graph_attr=graph_attr,
    outformat="png",
):
    clients = Users("Inference Clients")
    ingress = Ingress("Ingress /\nRoute")

    with Cluster("Kubernetes Cluster"):

        with Cluster("KServe Control Plane"):
            kserve = Deployment("KServe\nController")
            isvc = ConfigMap("InferenceService\nCRD")

        with Cluster("MLServer Pod"):
            rest = Service("REST :8080")
            grpc = Service("gRPC :8081")
            metrics_svc = Service("Metrics :8082")

            with Cluster("MLServer Container"):
                with Cluster("Supervisor Process"):
                    supervisor = Pod("FrontendSupervisor\n(shared pool owner)")

                with Cluster("Frontend Processes × N\n(SO_REUSEPORT)"):
                    frontends = Pod("REST · gRPC ·\nmetrics · dispatch")

                with Cluster("Shared Inference Pool"):
                    workers = Pod("Inference Workers\n(one copy per model)")

                with Cluster("Shipped Runtimes"):
                    sklearn = Pod("sklearn")
                    xgboost = Pod("xgboost")
                    lightgbm = Pod("lightgbm")
                    onnx = Pod("onnx")

        with Cluster("Storage"):
            model_pvc = PVC("Model Store\n(PVC)")
            s3 = Storage("S3 / MinIO\n(remote)")

        with Cluster("Observability"):
            prom = Prometheus("Prometheus")
            grafana = Grafana("Grafana")

        with Cluster("Security"):
            allowlist = ConfigMap("trusted-runtimes.json\n(/etc/mlserver/)")

    # Client → Ingress → Server
    clients >> Edge(label="HTTP / gRPC", color="#4A90D9") >> ingress
    ingress >> Edge(color="#4A90D9") >> rest
    ingress >> Edge(color="#4A90D9") >> grpc

    # Services → frontends (kernel-level SO_REUSEPORT balancing)
    rest >> Edge(color="#2C5F8A") >> frontends
    grpc >> Edge(color="#2C5F8A") >> frontends

    # Frontends → workers (ZeroMQ inference traffic)
    frontends >> Edge(label="ZeroMQ", color="#C0392B") >> workers

    # Supervisor coordination
    supervisor >> Edge(label="lifecycle (pipes)", color="#B58900") >> frontends
    supervisor >> Edge(color="#B58900") >> workers

    # Workers → Runtimes
    workers >> Edge(color="#7ED321", style="dashed") >> sklearn
    workers >> Edge(color="#7ED321", style="dashed") >> xgboost
    workers >> Edge(color="#7ED321", style="dashed") >> lightgbm
    workers >> Edge(color="#7ED321", style="dashed") >> onnx

    # Supervisor → Storage (artifact loading coordinates the shared pool)
    supervisor >> Edge(label="load artifacts", color="#F5A623") >> model_pvc
    supervisor >> Edge(label="fetch remote", color="#F5A623", style="dashed") >> s3

    # Observability
    prom >> Edge(label="scrape :8082", color="#D0021B") >> metrics_svc
    metrics_svc >> Edge(color="#D0021B") >> frontends
    prom >> Edge(color="#D0021B") >> grafana

    # KServe
    kserve >> Edge(label="reconcile", color="#8B572A", style="dotted") >> isvc
    isvc >> Edge(label="manage pod", color="#8B572A", style="dotted") >> supervisor

    # Security
    allowlist >> Edge(label="mount", color="#BD10E0", style="dotted") >> supervisor

    # Response path
    frontends >> Edge(color="#7ED321", style="dashed") >> rest
    rest >> Edge(color="#7ED321", style="dashed") >> ingress
    ingress >> Edge(color="#7ED321", style="dashed") >> clients
