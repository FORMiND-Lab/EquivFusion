FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ccache \
    ninja-build \
    git \
    curl \
    ca-certificates \
    meson \
    pkg-config \
    libgmp-dev \
    libmpfr-dev \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libreadline-dev \
    libz3-dev \
    z3 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" \
      | gpg --dearmor -o /etc/apt/keyrings/sbt.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/sbt.gpg] https://repo.scala-sbt.org/scalasbt/debian all main" \
      > /etc/apt/sources.list.d/sbt.list \
    && echo "deb [signed-by=/etc/apt/keyrings/sbt.gpg] https://repo.scala-sbt.org/scalasbt/debian /" \
      > /etc/apt/sources.list.d/sbt_old.list \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    sbt \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app/EquivFusion

WORKDIR /app/EquivFusion

RUN mkdir build && cd build \
    && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    && ninja && ninja install_solvers

RUN python3 -m venv torch_mlir_venv \
    && torch_mlir_venv/bin/python -m pip install --upgrade pip \
    && torch_mlir_venv/bin/pip install --pre torch-mlir torchvision \
    --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
    -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
    
ENV PATH="/app/EquivFusion/build/bin:/app/EquivFusion/circt_prebuild/bin:$PATH"

CMD ["/bin/bash"]


