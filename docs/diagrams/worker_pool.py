"""MLServer Worker Pool Architecture — Parallel Inference via Multiprocessing."""

import os

from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.compute import Server
from diagrams.onprem.client import Users
from diagrams.programming.framework import FastAPI
from diagrams.generic.compute import Rack
from diagrams.generic.storage import Storage

graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.8",
    "nodesep": "0.8",
    "ranksep": "1.0",
}

with Diagram(
    "MLServer — Worker Pool Architecture",
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker_pool"),
    show=False,
    direction="TB",
    graph_attr=graph_attr,
    outformat="png",
):
    clients = Users("Inference Requests")

    with Cluster(
        "Main Process", graph_attr={"bgcolor": "#E8F4FD", "pencolor": "#4A90D9"}
    ):
        rest = FastAPI("REST :8080")
        grpc = Server("gRPC :8081")
        dataplane = Rack("DataPlane")
        parallel_model = Rack("ParallelModel\n(proxy)")
        dispatcher = Rack("Dispatcher")

        with Cluster("Request / Response Queues"):
            req_queue = Storage("Request Queue\n(multiprocessing)")
            resp_queue = Storage("Response Queue\n(multiprocessing)")

    with Cluster(
        "Worker Process 1", graph_attr={"bgcolor": "#F0FFF0", "pencolor": "#7ED321"}
    ):
        worker1_loop = Server("asyncio event loop")
        worker1_model = Rack("MLModel copy 1\n(sklearn / xgboost / ...)")
        worker1_metrics = Storage(".metrics/\n(Prometheus dir)")

    with Cluster(
        "Worker Process 2", graph_attr={"bgcolor": "#F0FFF0", "pencolor": "#7ED321"}
    ):
        worker2_loop = Server("asyncio event loop")
        worker2_model = Rack("MLModel copy 2\n(sklearn / xgboost / ...)")
        worker2_metrics = Storage(".metrics/\n(Prometheus dir)")

    with Cluster(
        "Worker Process N", graph_attr={"bgcolor": "#F0FFF0", "pencolor": "#7ED321"}
    ):
        worker3_loop = Server("asyncio event loop")
        worker3_model = Rack("MLModel copy N\n(sklearn / xgboost / ...)")
        worker3_metrics = Storage(".metrics/\n(Prometheus dir)")

    model_store = Storage("Model Artifacts\n(filesystem / PVC)")

    # Client → Transport → DataPlane
    clients >> Edge(color="#4A90D9", style="bold") >> rest
    clients >> Edge(color="#2C5F8A", style="bold") >> grpc
    rest >> Edge(color="#4A90D9") >> dataplane
    grpc >> Edge(color="#2C5F8A") >> dataplane

    # DataPlane → ParallelModel → Dispatcher
    dataplane >> Edge(label="infer()", color="#7ED321") >> parallel_model
    parallel_model >> Edge(label="predict()", color="#7ED321") >> dispatcher

    # Dispatcher → Queues
    (
        dispatcher
        >> Edge(label="enqueue request", color="#F5A623", style="bold")
        >> req_queue
    )

    # Queues → Workers (fan-out)
    req_queue >> Edge(label="dequeue", color="#F5A623") >> worker1_loop
    req_queue >> Edge(label="dequeue", color="#F5A623") >> worker2_loop
    req_queue >> Edge(label="dequeue", color="#F5A623") >> worker3_loop

    # Workers process
    worker1_loop >> Edge(color="#7ED321") >> worker1_model
    worker2_loop >> Edge(color="#7ED321") >> worker2_model
    worker3_loop >> Edge(color="#7ED321") >> worker3_model

    # Workers → Response Queue (fan-in)
    worker1_loop >> Edge(label="result", color="#D0021B") >> resp_queue
    worker2_loop >> Edge(label="result", color="#D0021B") >> resp_queue
    worker3_loop >> Edge(label="result", color="#D0021B") >> resp_queue

    # Response back to dispatcher
    (
        resp_queue
        >> Edge(label="return result", color="#D0021B", style="bold")
        >> dispatcher
    )

    # Workers → Metrics
    worker1_model >> Edge(style="dotted", color="#9B59B6") >> worker1_metrics
    worker2_model >> Edge(style="dotted", color="#9B59B6") >> worker2_metrics
    worker3_model >> Edge(style="dotted", color="#9B59B6") >> worker3_metrics

    # Workers load model artifacts
    (
        worker1_model
        >> Edge(label="load()", color="#8B572A", style="dashed")
        >> model_store
    )
    worker2_model >> Edge(color="#8B572A", style="dashed") >> model_store
    worker3_model >> Edge(color="#8B572A", style="dashed") >> model_store
