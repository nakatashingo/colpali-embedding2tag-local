FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# 最小限の依存関係
RUN apt update -y && apt install -y \
    curl ca-certificates libsndfile1-dev libgl1 git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# uv インストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# /opt/venv に Python 3.12 の venv を作成
RUN uv venv /opt/venv --python 3.12

# uv sync / uv add がこの venv を使うよう設定
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pyproject.toml をコピーして依存関係をインストール
COPY workspace/pyproject.toml /workspace/
RUN uv sync

CMD ["bash"]
