# AMD ROCm (RDNA4 友好) Dockerfile
#
# 说明：
# - RDNA4 的支持依赖 ROCm / PyTorch 版本；这里默认使用 rocm/pytorch:latest，
#   需要可复现时请在 build 时用 --build-arg BASE_IMAGE=... 固定一个存在的 tag。
# - 不要在容器内 pip 安装 torch/torchvision/torchaudio（会覆盖 ROCm 版 torch）。
# - flash-attn / bitsandbytes 通常是 CUDA 专用；本镜像会自动跳过它们。

ARG BASE_IMAGE=rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0
FROM ${BASE_IMAGE}

WORKDIR /workspace

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_NO_CACHE_DIR=1 \
	HF_HUB_ENABLE_HF_TRANSFER=1 \
	PYTHONPATH=/workspace

RUN apt-get update \
	&& apt-get install -y --no-install-recommends git vim ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

# 过滤 requirements，避免覆盖 ROCm 版 torch，并跳过 CUDA 专用包
RUN python - <<'PY'
import re
from pathlib import Path

src = Path('requirements.txt').read_text(encoding='utf-8').splitlines()
out = []
skip = { 'torch', 'torchvision', 'torchaudio', 'flash-attn', 'flash_attn', 'bitsandbytes' }

name_re = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
for line in src:
	raw = line.strip()
	if not raw or raw.startswith('#'):
		continue
	# 保留 -r / --extra-index-url 等 pip 指令
	if raw.startswith('-'):
		out.append(line)
		continue
	m = name_re.match(raw)
	pkg = (m.group(1) if m else '').lower()
	if pkg in skip:
		continue
	out.append(line)

Path('/tmp/requirements.filtered.txt').write_text('\n'.join(out) + '\n', encoding='utf-8')
PY

RUN python -m pip install -U pip \
	&& python -m pip install -r /tmp/requirements.filtered.txt

RUN python -m spacy download en_core_web_sm

# RDNA4 兼容性：默认不强制 HSA 覆盖。
# 如果你的 ROCm 版本暂时不识别新卡，可在 docker run 时按需设置（示例）：
#   -e HSA_OVERRIDE_GFX_VERSION=xx.x.x

CMD ["/bin/bash"]

