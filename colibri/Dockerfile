FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git python3 ca-certificates && rm -rf /var/lib/apt/lists/*
ARG COLIBRI_REVISION=e479104ae01bdc40ce81e675299b1d230891b321
ARG CUDA_ARCH=sm_120
WORKDIR /opt/colibri
RUN git init . && git remote add origin https://github.com/JustVugg/colibri.git && git fetch --depth 1 origin ${COLIBRI_REVISION} && git checkout --detach FETCH_HEAD
RUN make -C c qwen36 CUDA=1 CUDA_ARCH=${CUDA_ARCH} ARCH=x86-64-v3 -j8
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
ENTRYPOINT ["python3", "/opt/colibri/c/coli"]
