# ------------------------------------------------------------------------------------------------------------
# Base stage — environment setup with caching optimization
# ------------------------------------------------------------------------------------------------------------
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04  AS base

USER root

# Install system dependencies (split to preserve caching)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget make g++ libboost-filesystem-dev libboost-system-dev \
        xutils-dev libxss1 xscreensaver xscreensaver-gl-extra xvfb python3 python3-dev python3-venv && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Copy environment-related files first (for caching)
COPY pyproject.toml ./

COPY mol_gen_docking ./mol_gen_docking
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --ignore-requires-python meeko==0.6.1 && \
    pip install ProDy uvicorn ringtail openbabel-wheel && \
    pip install pytdc==1.1.14 --no-deps
RUN pip install .


# ------------------------------------------------------------------------------------------------------------
# Builder stage — build AutoDock-GPU efficiently
# ------------------------------------------------------------------------------------------------------------
FROM base AS autodock-builder
USER root

# Install build dependencies (cacheable)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/build
RUN git clone --depth 1 https://github.com/ccsb-scripps/AutoDock-GPU.git

ENV GPU_INCLUDE_PATH=/usr/local/cuda/include
ENV GPU_LIBRARY_PATH=/usr/local/cuda/lib64
RUN cd AutoDock-GPU && make DEVICE=CUDA

# Store compiled binaries in a well-defined location
RUN cp -r AutoDock-GPU /opt/AutoDock-GPU


# ------------------------------------------------------------------------------------------------------------
# Final runtime image
# ------------------------------------------------------------------------------------------------------------
FROM base AS final
USER root

# Copy built binaries
COPY --from=autodock-builder /opt/AutoDock-GPU /opt/AutoDock-GPU
ENV PATH="/opt/AutoDock-GPU/bin:${PATH}"

RUN set -eux; \
    # ADFRsuite
    wget -O /tmp/ADFRsuite.tar.gz https://ccsb.scripps.edu/adfr/download/1038/; \
    tar -xzf /tmp/ADFRsuite.tar.gz -C /opt/; \
    AD_DIR=$(ls -d /opt/ADFRsuite_* 2>/dev/null || true); \
    if [ -n "$AD_DIR" ]; then mv "$AD_DIR" /opt/ADFRsuite; fi; \
    if [ -d /opt/ADFRsuite ]; then cd /opt/ADFRsuite && echo "Y" | ./install.sh -d . -c 0; fi; \
    rm -rf /tmp/ADFRsuite.tar.gz; \
    # Vina
    wget -O /tmp/vina.tgz --no-check-certificate https://vina.scripps.edu/wp-content/uploads/sites/55/2020/12/autodock_vina_1_1_2_linux_x86.tgz; \
    tar -xzf /tmp/vina.tgz -C /opt/; \
    mv /opt/autodock_vina_1_1_2_linux_x86/bin/* /usr/local/bin/ || true; \
    rm -rf /tmp/vina.tgz /opt/autodock_vina_1_1_2_linux_x86

# expose ADFRsuite bin (if present) and common lib path (some ADFR tools use their own libs)
ENV PATH="/opt/ADFRsuite/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/ADFRsuite/lib:${LD_LIBRARY_PATH:-}"

WORKDIR /
CMD ["/bin/bash"]
