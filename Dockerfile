FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OSHA_ENVIRONMENT_ID=kubernetes \
    OSHA_INSTANCE_NAME=osha

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt

COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY orchestration-ai-data/ /app/orchestration-ai-data/
COPY artifacts/ /app/artifacts/
COPY models/ /app/models/

RUN mkdir -p \
        /app/artifacts \
        /app/checkpoints \
        /app/models \
        /app/logs \
        /app/orchestration-ai-data \
    && addgroup --system osha \
    && adduser --system --ingroup osha --uid 10001 osha \
    && chown -R osha:osha /app

USER osha

EXPOSE 8088

CMD ["/bin/bash", "/app/scripts/runtime/run.sh"]