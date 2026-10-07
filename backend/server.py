"""Low-latency FasterLivePortrait websocket service.

Protocol:
  1. client connects to /ws/liveportrait
  2. client sends the source portrait as JPEG/PNG binary
  3. server sends {"type":"source_ready"}
  4. client sends one webcam JPEG at a time; server returns one rendered JPEG

Only one frame is in flight, providing natural backpressure and low latency.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

ROOT = Path(os.environ.get("FASTER_LIVEPORTRAIT_ROOT", "/opt/FasterLivePortrait"))
CHECKPOINTS = Path(os.environ.get("FLIP_CHECKPOINT_DIR", ROOT / "checkpoints"))
CONFIG = Path(os.environ.get("FLIP_CONFIG", ROOT / "configs/trt_mp_infer.yaml"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "92"))
pipe: FasterLivePortraitPipeline | None = None
pipe_lock = asyncio.Lock()


def _patch_paths(cfg):
    for group in (cfg.models, cfg.animal_models):
        for model in group.values():
            if "model_path" not in model:
                continue
            paths = model.model_path if isinstance(model.model_path, list) else [model.model_path]
            patched = [str(p).replace("./checkpoints", str(CHECKPOINTS)) for p in paths]
            model.model_path = patched if isinstance(model.model_path, list) else patched[0]
    cfg.infer_params.mask_crop_path = str(ROOT / "assets/mask_template.png")
    return cfg


def _ensure_models(cfg):
    missing = []
    for model in cfg.models.values():
        path = model.get("model_path")
        if not path:
            continue
        for item in path if isinstance(path, list) else [path]:
            if not Path(item).exists():
                onnx = Path(str(item).removesuffix(".trt") + ".onnx")
                if not onnx.exists():
                    missing.append(onnx)
    if missing:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        subprocess.run(["huggingface-cli", "download", "warmshao/FasterLivePortrait", "--local-dir", str(CHECKPOINTS)], check=True)
    for model in cfg.models.values():
        path = model.get("model_path")
        if not path:
            continue
        for engine in path if isinstance(path, list) else [path]:
            engine = Path(engine)
            if engine.exists():
                continue
            onnx = Path(str(engine).removesuffix(".trt") + ".onnx")
            subprocess.run(["python", str(ROOT / "scripts/onnx2trt.py"), "-o", str(onnx)], cwd=ROOT, check=True)


def _load_pipeline():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the low-latency TensorRT renderer")
    cfg = _patch_paths(OmegaConf.load(CONFIG))
    _ensure_models(cfg)
    cfg.infer_params.flag_pasteback = True
    cfg.infer_params.flag_stitching = True
    cfg.infer_params.flag_relative_motion = True
    return FasterLivePortraitPipeline(cfg=cfg, is_animal=False)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipe
    pipe = await asyncio.to_thread(_load_pipeline)
    yield
    pipe = None


app = FastAPI(title="FaceMirror TensorRT Renderer", lifespan=lifespan)


@app.get("/health")
async def health():
    return JSONResponse({
        "ready": pipe is not None,
        "backend": "FasterLivePortrait/TensorRT",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


def _decode(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _prepare_source(data: bytes):
    assert pipe is not None
    image = _decode(data)
    if image is None:
        raise ValueError("Could not decode source portrait")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        path = handle.name
    try:
        cv2.imwrite(path, image)
        if not pipe.prepare_source(path, realtime=True):
            raise ValueError("No clear face was found in the source portrait")
    finally:
        Path(path).unlink(missing_ok=True)


def _render(data: bytes, first_frame: bool):
    assert pipe is not None
    frame = _decode(data)
    if frame is None:
        raise ValueError("Could not decode driver frame")
    _, out_crop, out_org, _ = pipe.run(frame, pipe.src_imgs[0], pipe.src_infos[0], first_frame=first_frame)
    output = out_org if out_org is not None else out_crop
    if output is None:
        raise ValueError("No face found in driver frame")
    # FasterLivePortrait outputs RGB; OpenCV JPEG encoding expects BGR.
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode output frame")
    return encoded.tobytes()


@app.websocket("/ws/liveportrait")
async def liveportrait_socket(ws: WebSocket):
    await ws.accept()
    first_frame = True
    try:
        if pipe is None:
            await ws.send_json({"type": "error", "message": "GPU renderer is not ready"})
            return
        source = await ws.receive_bytes()
        async with pipe_lock:
            await asyncio.to_thread(_prepare_source, source)
            await ws.send_json({"type": "source_ready", "backend": "TensorRT"})
            while True:
                driver = await ws.receive_bytes()
                started = time.perf_counter()
                output = await asyncio.to_thread(_render, driver, first_frame)
                first_frame = False
                await ws.send_bytes(output)
                # Timing is sent after the frame so rendering can begin immediately.
                await ws.send_json({"type": "timing", "ms": round((time.perf_counter() - started) * 1000, 1)})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
