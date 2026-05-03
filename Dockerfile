# syntax=docker/dockerfile:1
#
# Install basic packages from OS distro
#
FROM ubuntu:22.04 AS base

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl libnuma-dev gnupg wget git nano \
        sudo libelf1 kmod file build-essential python3-dev \
        software-properties-common sqlite3 libsqlite3-dev libfmt-dev \
        libmsgpack-dev libsuitesparse-dev libdw-dev cython3 ibverbs-utils openmpi-bin libopenmpi-dev \
        libpci-dev

# Python
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    if ! python3 --version | grep -q ${PYTHON_VERSION} ; then \
        add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
        apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev \
            python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-lib2to3 python-is-python3 ; \
    fi

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

RUN wget -nv -O /tmp/cmake-3.26.4-linux-x86_64.tar.gz https://cmake.org/files/v3.26/cmake-3.26.4-linux-x86_64.tar.gz && \
    tar zfx /tmp/cmake-3.26.4-linux-x86_64.tar.gz -C /opt/ && \
    mv /opt/cmake-3.26.4-linux-x86_64 /opt/cmake-3.26.4 && \
    rm -f /tmp/cmake-3.26.4-linux-x86_64.tar.gz

ENV PATH=/opt/cmake-3.26.4/bin:$PATH

FROM base AS rocm_deb

ARG ROCM_VERSION=7.2
ARG AMDGPU_VERSION=30.30
ARG GPU_ARCH=gfx942
ARG PYTHON_VERSION=3.12

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ jammy main" | tee /etc/apt/sources.list.d/rocm.list && \
    printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu jammy main" | tee /etc/apt/sources.list.d/amdgpu.list && \
    printf "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n" > /etc/apt/preferences.d/artifactory-pin-600 && \
    apt-get update && \
    apt-get install -y rocm && \
    find /opt/rocm/lib -type f -name '*gfx*' | grep -Ev "${GPU_ARCH}" | xargs rm -f && \
    find /opt/rocm/lib/hipblaslt/library -type f -name '*gfx*' | grep -Ev "${GPU_ARCH}" | xargs rm -f && \
    find /opt/rocm/lib/rocblas/library -type f -name '*gfx*' | grep -Ev "${GPU_ARCH}" | xargs rm -f && \
    find /opt/rocm/share/miopen/db -type f -name '*gfx*' | grep -Ev "${GPU_ARCH}" | xargs rm -f && \
    sqlite3 /opt/rocm/lib/rocfft/rocfft_kernel_cache.db "delete from cache_v1 where arch != '${GPU_ARCH}' ; vacuum"

# ROCm meta-package can disturb/remove python alternatives; reinstall and pin interpreter.
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python-is-python3 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 100 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && test -x /usr/bin/python3 && /usr/bin/python3 --version

# Shared venv: parallel stages install here so COPY --from=... /opt/venv works
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel

ENV ROCM_HOME=/opt/rocm
ENV CPLUS_INCLUDE_PATH=/opt/rocm/include
ENV LD_LIBRARY_PATH=/opt/rocm/lib
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
ENV GPU_ARCH_LIST=$GPU_ARCH

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /tmp && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl -O torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/apex-1.9.0+rocm7.2.0.gite37ed124-cp312-cp312-linux_x86_64.whl -O apex-1.9.0+rocm7.2.0.gite37ed124-cp312-cp312-linux_x86_64.whl && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0+rocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl -O torchaudio-2.9.0+rocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl -O torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl && \
    wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1+rocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl -O triton-3.5.1+rocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl && \
    python3 -m pip install \
        torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl \
        apex-1.9.0+rocm7.2.0.gite37ed124-cp312-cp312-linux_x86_64.whl \
        torchaudio-2.9.0+rocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
        torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
        triton-3.5.1+rocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl && \
    rm -f /tmp/*.whl

# --------------------------------------------------------------------
# Stage 1: MORI — parallel
# --------------------------------------------------------------------
FROM rocm_deb AS build_mori
ARG MORI_COMMIT="v1.1.0"


RUN echo "========== [Parallel] Building MORI ==========" && \
    git clone https://github.com/ROCm/mori.git /app/mori && \
    cd /app/mori && \
    git checkout $MORI_COMMIT && \
    pip install .

# --------------------------------------------------------------------
# Stage 2: RCCL — parallel
# --------------------------------------------------------------------
FROM rocm_deb AS build_rccl
ARG RCCL_REPO="https://github.com/ROCm/rccl.git"
ARG RCCL_BRANCH="29e1567b95e28823b0beb1a988adc587bfab5b4f"

RUN echo "========== [Parallel] Building RCCL ==========" && \
    pip install cmake && \
    git clone "$RCCL_REPO" /app/rccl && \
    cd /app/rccl && \
    git checkout "$RCCL_BRANCH" && \
    ./install.sh -p --amdgpu_targets=$GPU_ARCH_LIST

# --------------------------------------------------------------------
# Stage 3: Triton — parallel
# --------------------------------------------------------------------
FROM rocm_deb AS build_triton

RUN echo "========== [Parallel] Building Triton ==========" && \
    pip uninstall -y triton && \
    git clone --depth=1 --branch release/internal/3.5.x \
        https://github.com/ROCm/triton.git /triton-test && \
    cd /triton-test && \
    pip install -r python/requirements.txt && \
    pip install filecheck && \
    MAX_JOBS=64 pip --retries=10 --default-timeout=60 install .

# --------------------------------------------------------------------
# Stage 4: Aiter — parallel
# --------------------------------------------------------------------
FROM rocm_deb AS build_aiter
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_COMMIT="HEAD"
ARG PREBUILD_KERNELS=1
ARG MAX_JOBS

RUN pip install --upgrade setuptools_scm
RUN echo "========== [Parallel] Building Aiter ==========" && \
    git clone --depth 1 $AITER_REPO /app/aiter-test && \
    cd /app/aiter-test && \
    pip install -r requirements.txt && \
    git checkout $AITER_COMMIT && \
    git submodule sync && git submodule update --init --recursive && \
    MAX_JOBS=$MAX_JOBS PREBUILD_KERNELS=$PREBUILD_KERNELS \
    GPU_ARCHS=$GPU_ARCH_LIST python setup.py develop

# --------------------------------------------------------------------
# Stage 5: Final merge — collect all build artifacts + install ATOM
# --------------------------------------------------------------------
FROM rocm_deb AS atom_image
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="HEAD"

# MORI venv first (pip install lm-eval before COPY would be wiped when replacing /opt/venv)
COPY --from=build_mori /opt/venv /opt/venv
COPY --from=build_mori /app/mori /app/mori
RUN pip show mori || true

RUN pip install lm-eval[api]

# RCCL: install .deb from build stage
COPY --from=build_rccl /app/rccl/build/release/*.deb /tmp/rccl/
RUN DEBIAN_FRONTEND=noninteractive dpkg -i --force-all /tmp/rccl/*.deb && \
    rm -rf /tmp/rccl

# Triton: copy package from build stage into current venv
# Use RUN --mount to avoid COPY glob issues, and preserve mori already in venv
RUN --mount=from=build_triton,source=/opt/venv/lib/python3.12/site-packages,target=/triton-pkgs \
    cp -a /triton-pkgs/triton /opt/venv/lib/python3.12/site-packages/ && \
    cp -a /triton-pkgs/triton-*.dist-info /opt/venv/lib/python3.12/site-packages/ && \
    pip show triton

# Aiter: copy compiled source tree + re-register editable install
# (pip install -e creates egg-link automatically, no need to COPY them)
COPY --from=build_aiter /app/aiter-test /app/aiter-test
RUN cd /app/aiter-test && pip install -e . --no-build-isolation && \
    pip show amd-aiter

# RTL (rocm-trace-lite): lightweight GPU kernel profiler (~250KB, no build deps)
RUN pip install rocm-trace-lite && \
    rtl --version || true

# ATOM: lightweight install (no compilation needed)
# CACHEBUST invalidates only this layer so parallel stages stay cached
ARG CACHEBUST=1
RUN git clone $ATOM_REPO /app/ATOM && \
    cd /app/ATOM && \
    git checkout $ATOM_COMMIT && \
    pip install -e .
RUN pip show atom || true

FROM atom_image AS vllm_image

ARG VENV_PYTHON="/opt/venv/bin/python"
ARG VLLM_REPO="https://github.com/vllm-project/vllm.git"
ARG VLLM_COMMIT="b31e9326a7d9394aab8c767f8ebe225c65594b60"
ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="HEAD"
ARG MAX_JOBS=16
ARG INSTALL_LM_EVAL=1
ARG INSTALL_FASTSAFETENSORS=1
ARG CACHEBUST=1

ENV VLLM_TARGET_DEVICE=rocm
ENV MAX_JOBS=${MAX_JOBS}
ENV CMAKE_MAKE_PROGRAM=/usr/local/bin/ninja
LABEL com.rocm.atom.vllm_commit="${VLLM_COMMIT}"

RUN python3 -c "import os, triton, pathlib, torch; \
    s = str(pathlib.Path(triton.__file__).resolve().parent.parent); \
    t = os.path.join(os.path.dirname(torch.__file__), 'lib'); \
    open('/root/atom_vllm_site_pkgs', 'w').write(s); \
    open('/root/atom_vllm_torch_lib', 'w').write(t)"

ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib/python3.12/dist-packages/torch/lib

RUN echo "========== [vLLM OOT 1/6] Clone vLLM ==========" && \
    rm -rf /app/vllm && \
    git clone ${VLLM_REPO} /app/vllm && \
    cd /app/vllm && \
    git checkout ${VLLM_COMMIT} && \
    git submodule update --init --recursive && \
    echo "vLLM commit:" && \
    git rev-parse HEAD

RUN echo "========== [vLLM OOT 2/6] vLLM ROCm deps ==========" && \
    cd /app/vllm && \
    python3 -m pip install --upgrade pip && \
    sed -i -e '/^transformers[[:space:]]/d' requirements/common.txt && \
    sed -i -e '/xgrammar/d' -e '/compressed-tensors/d' requirements/common.txt && \
    python3 -m pip install --no-deps xgrammar==0.1.29 compressed-tensors==0.13.0 loguru && \
    sed -i -e '/peft/d' -e '/tensorizer/d' -e '/runai/d' -e '/timm/d' requirements/rocm.txt && \
    python3 -m pip install --no-deps peft tensorizer==2.10.1 runai-model-streamer[s3,gcs]==0.15.3 "timm>=1.0.17" && \
    python3 -m pip install -r requirements/rocm.txt

RUN echo "========== [vLLM OOT 3/6] amd-smi wheel ==========" && \
    cd /opt/rocm/share/amd_smi && \
    python3 -m pip wheel . --wheel-dir=dist && \
    python3 -m pip install dist/*.whl

RUN echo "========== [vLLM OOT 5/6] Build vLLM wheel ==========" && \
    cd /app/vllm && \
    VLLM_TARGET_DEVICE=rocm python3 setup.py clean --all && \
    MAX_JOBS="${MAX_JOBS}" VLLM_TARGET_DEVICE=rocm python3 setup.py bdist_wheel --dist-dir=/tmp/vllm-wheels && \
    ls -lh /tmp/vllm-wheels

RUN echo "========== [vLLM OOT 6/6] Install vLLM and extras ==========" && \
    cd /app/vllm && \
    python3 -m pip uninstall -y vllm 2>/dev/null || true && \
    python3 -m pip install /tmp/vllm-wheels/*.whl && \
    python3 -m pip install uvloop && \
    if [ "${INSTALL_LM_EVAL}" = "1" ]; then "${VENV_PYTHON}" -m pip install "lm-eval[api]"; else echo "Skip lm-eval install"; fi && \
    if [ "${INSTALL_FASTSAFETENSORS}" = "1" ]; then python3 -m pip install "git+https://github.com/foundation-model-stack/fastsafetensors.git"; else echo "Skip fastsafetensors"; fi && \
    python3 -c "import glob, os, torch; print(f'torch.version.hip: {torch.version.hip}'); print(f'torch.version.cuda: {torch.version.cuda}'); torch_lib_dir=os.path.join(os.path.dirname(torch.__file__), 'lib'); print(f'torch lib dir: {torch_lib_dir}'); print(f'libtorch_hip: {glob.glob(os.path.join(torch_lib_dir, \"libtorch_hip.so*\"))}'); assert torch.version.hip is not None" && \
    python3 -m pip show vllm torch triton torchvision torchaudio amdsmi atom || true

RUN echo "========== [vLLM-ATOM] Validate vision/audio wheels ==========" && \
    python3 -c "import torch, torchvision, torchaudio; from torchvision.transforms import InterpolationMode; from transformers.models.auto.image_processing_auto import get_image_processor_config; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'InterpolationMode: {InterpolationMode.BILINEAR}'); print(f'get_image_processor_config: {get_image_processor_config.__name__}')"

FROM vllm_image AS final

RUN --mount=type=bind,from=ubuntu:22.04,source=/,target=/tmp \
    cp /tmp/etc/apt/apt.conf.d/docker-clean /etc/apt/apt.conf.d/docker-clean

ENV DEBIAN_FRONTEND=

WORKDIR /root

CMD ["/bin/bash"]
