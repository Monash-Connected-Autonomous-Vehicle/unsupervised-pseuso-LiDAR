FROM dvcorg/cml-py3

ARG python=3.8
ENV PYTHON_VERSION=${python}

ENV PROJECT=unsup-mono
ENV PYTORCH_VERSION=1.8.1
ENV TORCHVISION_VERSION=0.9.1
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.2
ENV NCCL_VERSION=2.7.8-1+cuda10.2

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-4.8 \
    git \
    curl \
    vim \
    wget

# Instal Python and pip
RUN if [[ "${PYTHON_VERSION}" == "3.8" ]]; then \
    apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install Pydata and other deps
RUN pip3 install numpy pandas matplotlib jupyter pyyaml \
    tqdm termcolor path.py pillow opencv-python-headless \
    onnx onnxruntime

# Install PyTorch
RUN pip3 install torch torchvision

# Install wandb
RUN pip3 install wandb==0.11.0 

WORKDIR /workspace
RUN git clone https://github.com/Monash-Connected-Autonomous-Vehicle/unsupervised-pseuso-LiDAR.git

CMD ls