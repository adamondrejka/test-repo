# Reality Engine Backend - Processing Pipeline
# Base image with CUDA support for GPU training

FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e ".[training]"

# Install nerfstudio (requires separate installation)
RUN pip install nerfstudio

# Copy application code (includes depth_splatfacto/ plugin)
COPY . .

# Re-install in editable mode to register entry points (depth-splatfacto plugin)
RUN pip install -e .

# Create directories for data
RUN mkdir -p /data/input /data/output /data/temp

# Set environment variables
ENV DATA_INPUT=/data/input
ENV DATA_OUTPUT=/data/output
ENV DATA_TEMP=/data/temp

# Expose port for potential API
EXPOSE 8000

# Default command
CMD ["python", "-m", "pipeline.process", "--help"]

# ============================================
# Development stage (without training deps)
# ============================================
FROM python:3.10-slim AS dev

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (without training)
COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install -e ".[dev]"

COPY . .

RUN mkdir -p /data/input /data/output /data/temp

ENV DATA_INPUT=/data/input
ENV DATA_OUTPUT=/data/output
ENV DATA_TEMP=/data/temp

CMD ["python", "-m", "pipeline.process", "--help"]
