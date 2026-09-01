"""MLServer Production Deployment Topology on Kubernetes."""

from diagrams import Cluster, Diagram, Edge
from diagrams.k8s.compute import Pod, Deployment
from diagrams.k8s.network import Service, Ingress
from diagrams.k8s.storage import PersistentVolumeClaim as PVC
from diagrams.k8s.podconfig import ConfigMap
from diagrams.onprem.monitoring import Prometheus, Grafana
from diagrams.onprem.queue import Kafka
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
    filename="/Users/imran/Projects/MLServer/docs/diagrams/deployment_topology",
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
                mlserver = Pod("MLServer\n(main process)")

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

        with Cluster("Messaging (optional)"):
            kafka = Kafka("Kafka Broker")

        with Cluster("Security"):
            allowlist = ConfigMap("trusted-runtimes.json\n(/etc/mlserver/)")

    # Client → Ingress → Server
    clients >> Edge(label="HTTP / gRPC", color="#4A90D9") >> ingress
    ingress >> Edge(color="#4A90D9") >> rest
    ingress >> Edge(color="#4A90D9") >> grpc

    # Services → MLServer
    rest >> Edge(color="#2C5F8A") >> mlserver
    grpc >> Edge(color="#2C5F8A") >> mlserver

    # MLServer → Runtimes
    mlserver >> Edge(color="#7ED321", style="dashed") >> sklearn
    mlserver >> Edge(color="#7ED321", style="dashed") >> xgboost
    mlserver >> Edge(color="#7ED321", style="dashed") >> lightgbm
    mlserver >> Edge(color="#7ED321", style="dashed") >> onnx

    # MLServer → Storage
    mlserver >> Edge(label="load artifacts", color="#F5A623") >> model_pvc
    mlserver >> Edge(label="fetch remote", color="#F5A623", style="dashed") >> s3

    # Observability
    prom >> Edge(label="scrape :8082", color="#D0021B") >> metrics_svc
    metrics_svc >> Edge(color="#D0021B") >> mlserver
    prom >> Edge(color="#D0021B") >> grafana

    # Kafka
    kafka >> Edge(label="consume /\nproduce", color="#9B59B6", style="bold") >> mlserver

    # KServe
    kserve >> Edge(label="reconcile", color="#8B572A", style="dotted") >> isvc
    isvc >> Edge(label="manage pod", color="#8B572A", style="dotted") >> mlserver

    # Security
    allowlist >> Edge(label="mount", color="#BD10E0", style="dotted") >> mlserver

    # Response path
    mlserver >> Edge(color="#7ED321", style="dashed") >> rest
    rest >> Edge(color="#7ED321", style="dashed") >> ingress
    ingress >> Edge(color="#7ED321", style="dashed") >> clients
