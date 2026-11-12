# gRPC route qualification — ROUTED_G_UNAVAILABLE

Tested from the in-cluster client pod against the frozen serving deployment
(mlserver-rfdetr, gRPC port 8081, plaintext h2c):

| Path | Result |
|---|---|
| Passthrough route + TLS | FAILED — TLS handshake `WRONG_VERSION_NUMBER`: the router forwards TLS to port 8081, where the server speaks plaintext; passthrough requires a TLS-serving backend |
| Edge route + TLS | FAILED — the route terminates TLS and forwards HTTP/1.1 to the backend; gRPC requires HTTP/2 end-to-end |
| Edge route + h2c (port 80) | FAILED — `UNAVAILABLE`: HTTP/2 prior-knowledge preface not accepted by the HTTP/1.1 route path |
| Direct service endpoint (control) | SUCCESS — 114,127-byte response, byte-exact (campaign reference) |

Conclusion: routed gRPC is unavailable in the current deployment topology.
The serving pod's gRPC port is plaintext; OpenShift routes require either a
TLS-serving backend (passthrough/reencrypt) or HTTP/2 backend forwarding
(edge provides HTTP/1.1 only). Enabling routed gRPC would require a
deployment change (in-pod TLS termination or an HTTP/2-capable ingress) —
outside this campaign's scope. gRPC results remain direct-predictor
controls, labeled as such.

Reproduce: oc exec <client-pod> -- python3 /store/grpc-route-test.py
