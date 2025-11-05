# Dockerfile for ocrd_party
ARG DOCKER_BASE_IMAGE=ocrd/core-cuda-torch:2024
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE

LABEL \
    maintainer="https://ocr-d.de/en/contact" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/CrazyCrud/ocrd_party" \
    org.label-schema.build-date=$BUILD_DATE \
    org.opencontainers.image.vendor="DFG-Funded Initiative for Optical Character Recognition Development" \
    org.opencontainers.image.title="ocrd_party" \
    org.opencontainers.image.description="OCR-D wrapper for the Party text recognizer" \
    org.opencontainers.image.source="https://github.com/CrazyCrud/ocrd_party" \
    org.opencontainers.image.documentation="https://github.com/CrazyCrud/ocrd_party" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.base.name=$DOCKER_BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Avoid HOME/.local/share (hard to predict USER here)
# so let XDG_DATA_HOME coincide with fixed system location
ENV XDG_DATA_HOME /usr/local/share

# Avoid the need for an extra volume for persistent resource user db
ENV XDG_CONFIG_HOME /usr/local/share/ocrd-resources

# Set up working directory
WORKDIR /build/ocrd_party

# Copy the entire project
COPY . .

# Ensure ocrd-tool.json symlink / copy exists at project root
COPY ocrd_party/ocrd-tool.json ./ocrd-tool.json

# Install system dependencies if needed (kept similar to YOLO image; you can trim if not necessary)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Prepackage ocrd-tool.json as ocrd-all-tool.json
RUN ocrd ocrd-tool ocrd-tool.json dump-tools > $(dirname $(ocrd bashlib filename))/ocrd-all-tool.json

# Prepackage ocrd-all-module-dir.json
RUN ocrd ocrd-tool ocrd-tool.json dump-module-dirs > $(dirname $(ocrd bashlib filename))/ocrd-all-module-dir.json

# Install the package and its dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Clean up build dependencies to reduce image size
RUN apt-get -y remove --auto-remove g++ && \
    apt-get clean && \
    rm -rf /build/ocrd_party

# Set the working directory for actual use
WORKDIR /data

# Create a volume for data
VOLUME /data

# Default command (can be overridden)
CMD ["ocrd-party-recognize", "--help"]
