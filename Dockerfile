FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" || \
    pip install flash-attn --no-build-isolation

RUN pip install \
    wandb==0.18.3 \
    tensorboardx \
    qwen_vl_utils \
    torchvision \
    git+https://github.com/huggingface/transformers.git

COPY ./src/open-r1-multimodal /workspace/src/open-r1-multimodal

WORKDIR /workspace/src/open-r1-multimodal
RUN pip install -e ".[dev]"
WORKDIR /workspace

RUN pip install vllm==0.7.2

ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]

