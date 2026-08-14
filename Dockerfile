FROM runpod/forge:3.3.0

USER root
RUN python3 -m venv /opt/serverless-venv \
    && /opt/serverless-venv/bin/pip install --no-cache-dir \
        "runpod>=1.7,<2" "httpx>=0.27,<1"

COPY handler.py /opt/comic-serverless/handler.py

ENV PYTHONUNBUFFERED=1 \
    DISABLE_AUTOLAUNCH=1 \
    FORGE_PORT=3001

CMD ["/opt/serverless-venv/bin/python", "-u", "/opt/comic-serverless/handler.py"]
