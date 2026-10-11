# Low-latency GPU deployment

The production path uses FasterLivePortrait with TensorRT on a persistent NVIDIA
GPU. The browser sends one webcam frame at a time over a backpressured WebSocket;
the server returns the generated portrait as soon as inference completes.

## Requirements

- Linux host
- Docker Engine with Compose
- NVIDIA driver and NVIDIA Container Toolkit
- NVIDIA GPU with at least 8 GB VRAM; 12 GB or more is recommended
- Approximately 15 GB free disk for the image, ONNX weights, and TensorRT engines

Verify Docker GPU access first:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Start

```bash
docker compose up --build
```

The first startup downloads FasterLivePortrait checkpoints and compiles
GPU-specific TensorRT engines. This can take 10–30 minutes. Engines and weights
are retained in the `liveportrait-models` volume. Subsequent starts are much
faster.

Open `http://HOST:8080`. For camera access outside localhost, terminate TLS at a
reverse proxy and forward to port 8080. The web client automatically tries the
TensorRT WebSocket first and falls back to local WebGPU only if the service is
unavailable.

## Architecture

```text
Camera -> JPEG WebSocket (one frame in flight)
       -> MediaPipe driver landmarks
       -> FasterLivePortrait motion extractor
       -> relative 3D implicit keypoints
       -> stitching / retargeting
       -> TensorRT warping + SPADE generator
       -> JPEG frame -> browser canvas
```

One-frame backpressure prevents a queue from accumulating, so latency remains
bounded by capture, network round trip, inference, and JPEG encode/decode time.
It does not claim mathematically zero latency. On a suitable local GPU/LAN, the
goal is responsive near-real-time motion rather than delayed frame playback.

## Health check

```bash
curl http://HOST:8080/health
```

A ready response includes the CUDA device and `FasterLivePortrait/TensorRT`.

## Development

Run the GPU service on port 8000, then:

```bash
GPU_BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

Vite proxies `/ws` and `/health` to the renderer.
