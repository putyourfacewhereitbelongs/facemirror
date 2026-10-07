FROM shaoguo/faster_liveportrait:v3

USER root
WORKDIR /opt
RUN rm -rf /opt/FasterLivePortrait && \
    git clone https://github.com/warmshao/FasterLivePortrait.git /opt/FasterLivePortrait && \
    git -C /opt/FasterLivePortrait checkout 8aad3602177547aaa5e4beec0c3ef5b7944e7a1f && \
    rm -rf /opt/FasterLivePortrait/.git && \
    pip install --no-cache-dir "fastapi>=0.115,<1" "uvicorn[standard]>=0.32,<1" "websockets>=14,<16"
COPY server.py /opt/FasterLivePortrait/facemirror_server.py
WORKDIR /opt/FasterLivePortrait
ENV FASTER_LIVEPORTRAIT_ROOT=/opt/FasterLivePortrait \
    FLIP_CHECKPOINT_DIR=/models \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8000
CMD ["uvicorn", "facemirror_server:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "16777216"]
