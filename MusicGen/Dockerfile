# Base Image: Ubuntu 20.04 with CUDA 11.8 development environment
# Using 20.04 as it has better support for older CUDA + python combinations via PPA
# and is more stable for building from source.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# Set non-interactive frontend for apt commands
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Python 3.10.13 from source
RUN wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz && \
    tar -xf Python-3.10.13.tgz && \
    cd Python-3.10.13 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.10.13 Python-3.10.13.tgz

# Make python3.10.13 the default python and python3
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3

# Verify Python version
RUN python --version

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .


RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies from requirements.txt
# Exclude packages already installed or handled separately
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# COPY frontend.HTML .

# Set default command to bash for interactive use
CMD ["uvicorn", "app_server_backend:app", "--host", "0.0.0.0", "--port", "8000"]