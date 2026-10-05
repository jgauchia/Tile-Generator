FROM debian:bookworm-slim

# Toolchain, generator dependencies and the runtime of the region web server phase
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libosmium2-dev \
        libgeos-dev \
        libgdal-dev \
        libbz2-dev \
        zlib1g-dev \
        libexpat1-dev \
        nlohmann-json3-dev \
        python3 \
        python3-shapely \
        osmium-tool \
    && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
