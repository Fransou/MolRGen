# ------------------------------------------------------------------------------------------------------------
# Base stage — environment setup with caching optimization
# ------------------------------------------------------------------------------------------------------------
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04  AS base

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget make g++ libboost-filesystem-dev libboost-system-dev \
        xutils-dev libxss1 xscreensaver xscreensaver-gl-extra xvfb \
        python3 python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

# 2. Set an environment variable to bypass the "Externally Managed Environment" error
# This is cleaner than adding --break-system-packages to every single line.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Copy environment-related files first (for caching)
COPY pyproject.toml ./
COPY mol_gen_docking ./mol_gen_docking
COPY test ./test
COPY data/properties.csv ./data/properties.csv

# 3. Install packages directly to the system Python
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install rdkit==2024.3.5 && \
    pip install --ignore-requires-python meeko==0.6.1 && \
    pip install ProDy uvicorn ringtail openbabel-wheel && \
    pip install ray==2.52.1 && \
    pip install fastapi-mcp && \
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
RUN git clone https://github.com/ccsb-scripps/AutoDock-GPU.git && \
    git checkout v1.6

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
ENV PATH="${PATH}:/opt/ADFRsuite/bin"
ENV LD_LIBRARY_PATH="/opt/ADFRsuite/lib:${LD_LIBRARY_PATH:-}"

WORKDIR /

# Expose port 8000 for the API server
EXPOSE 8000

# Create entry point script to handle command selection
RUN echo '#!/bin/bash' > /usr/local/bin/docker-entrypoint.sh && \
    echo 'if [ "$1" = "mcp-server" ]; then' >> /usr/local/bin/docker-entrypoint.sh && \
    echo '  exec BUFFER_TIME=1 PARSING_METHOD=none SERVER_MODE=batch python -m mol_gen_docking.server_mcp' >> /usr/local/bin/docker-entrypoint.sh && \
    echo 'elif [ "$1" = "api-server" ]; then' >> /usr/local/bin/docker-entrypoint.sh && \
    echo '  exec python -m mol_gen_docking.server' >> /usr/local/bin/docker-entrypoint.sh && \
    echo 'else' >> /usr/local/bin/docker-entrypoint.sh && \
    echo '  exec "$@"' >> /usr/local/bin/docker-entrypoint.sh && \
    echo 'fi' >> /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
{
  "smiles": [
    "CCCCC"
  ],
  "properties": [
    "sample_347546_model_0"
  ]
}
