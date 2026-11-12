import grpc

PASSTHROUGH = "mlserver-rfdetr-grpc-mlserver-perf.apps.syedali-cluster-pool-q7c2q.gcp.rh-ods.com"
EDGE = "mlserver-rfdetr-grpc-edge-mlserver-perf.apps.syedali-cluster-pool-q7c2q.gcp.rh-ods.com"
CA = open("/store/route-ca.pem", "rb").read()
RPC = "/inference.GRPCInferenceService/ModelInfer"
payload = open("/store/payloads/g-0.bin", "rb").read()

def attempt(label, target, creds):
    try:
        ch = grpc.secure_channel(target, creds)
        grpc.channel_ready_future(ch).result(timeout=10)
        method = ch.unary_unary(RPC)
        resp = method(payload, timeout=30)
        print(f"{label}: SUCCESS ({len(resp)} bytes)")
    except Exception as e:
        print(f"{label}: FAILED - {type(e).__name__}: {str(e)[:110]}")

attempt("passthrough + TLS ", f"{PASSTHROUGH}:443", grpc.ssl_channel_credentials(CA))
attempt("edge + TLS        ", f"{EDGE}:443", grpc.ssl_channel_credentials(CA))

for label, target in [("edge + h2c port80", f"{EDGE}:80"), ("direct (control) ", "mlserver-rfdetr.mlserver-perf.svc:8081")]:
    try:
        ch = grpc.insecure_channel(target)
        grpc.channel_ready_future(ch).result(timeout=10)
        resp = ch.unary_unary(RPC)(payload, timeout=30)
        print(f"{label}: SUCCESS ({len(resp)} bytes)")
    except Exception as e:
        print(f"{label}: FAILED - {type(e).__name__}: {str(e)[:110]}")
