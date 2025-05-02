ARG TAG
FROM nvidia/cuda:${TAG}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    \
    ca-certificates \
    curl \
    gnupg \
    procps \
    uuid-runtime \
    \
    && rm -rf /var/lib/apt/lists/*
