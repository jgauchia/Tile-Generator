#!/bin/bash

# Script to generate tiles from PBF using Docker
# Converts PBF -> GOL -> Tiles automatically

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════╗"
echo "║   PBF → GOL → Tiles (Docker)           ║"
echo "╚════════════════════════════════════════╝"
echo -e "${NC}"

# Initialize variables
FORCE_CLEAN=false
ZOOM_ARGS=""

# Parse all arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean-docker)
            FORCE_CLEAN=true
            shift
            ;;
        --zoom)
            ZOOM_ARGS="$ZOOM_ARGS --zoom $2"
            shift 2
            ;;
        --max-file-size)
            ZOOM_ARGS="$ZOOM_ARGS --max-file-size $2"
            shift 2
            ;;
        *)
            # Positional arguments
            if [ -z "$INPUT_PBF" ]; then
                INPUT_PBF=$1
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR=$1
            elif [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE=$1
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$INPUT_PBF" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <input.pbf> <output_dir> <config.json> [options]"
    echo ""
    echo "Options:"
    echo "  --clean-docker      Force rebuild of Docker image"
    echo "  --zoom N-M           Zoom range (e.g., 6-17)"
    echo "  --max-file-size KB   Maximum tile file size in KB"
    echo ""
    echo "Example:"
    echo "  $0 map.osm.pbf tiles features.json --zoom 6-17"
    echo "  $0 map.osm.pbf tiles features.json --clean-docker --zoom 12 --max-file-size 512"
    exit 1
fi

# Validate input
if [ ! -f "$INPUT_PBF" ]; then
    echo -e "${RED}✗ PBF file not found: $INPUT_PBF${NC}"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Get absolute paths
INPUT_ABS=$(realpath "$INPUT_PBF")
CONFIG_ABS=$(realpath "$CONFIG_FILE")
OUTPUT_ABS=$(realpath -m "$OUTPUT_DIR")

INPUT_NAME=$(basename "$INPUT_PBF")
CONFIG_NAME=$(basename "$CONFIG_FILE")
OUTPUT_NAME=$(basename "$OUTPUT_DIR")

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_ABS"
echo -e "${GREEN}✓ Output directory ready:${NC} $OUTPUT_ABS"
echo ""

echo -e "${GREEN}Input:${NC} $INPUT_NAME"
echo -e "${GREEN}Output:${NC} $OUTPUT_NAME"
echo -e "${GREEN}Config:${NC} $CONFIG_NAME"
echo ""
echo -e "${BLUE}Mounting paths:${NC}"
echo "  PBF: $INPUT_ABS -> /input/$INPUT_NAME"
echo "  Config: $CONFIG_ABS -> /config/$CONFIG_NAME"
echo "  Output: $OUTPUT_ABS -> /output"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    exit 1
fi

DOCKER_CMD="docker"
if [ "$EUID" -ne 0 ]; then
    DOCKER_CMD="sudo docker"
fi

echo -e "${YELLOW}Preparing Docker environment...${NC}"

# Create optimized Dockerfile
cat > /tmp/Dockerfile.tiles << 'EOF'
FROM ubuntu:24.04

# Install system dependencies with retries
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    wget ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies globally to avoid reinstalling every time
RUN pip3 install --no-cache-dir --break-system-packages \
    --default-timeout=100 \
    --retries=5 \
    osmium \
    geodesk \
    lz4 \
    tqdm \
    tabulate \
    shapely \
    psutil \
    ijson

WORKDIR /work
EOF

# Handle --clean-docker option
if [ "$FORCE_CLEAN" = true ]; then
    echo -e "${YELLOW}--clean-docker flag detected: will rebuild Docker image${NC}"
    IMAGE_EXISTS=$($DOCKER_CMD images tile-generator:latest -q 2>/dev/null || true)
    if [ -n "$IMAGE_EXISTS" ]; then
        echo "  Removing existing Docker image..."
        $DOCKER_CMD rmi tile-generator:latest > /dev/null 2>&1 || true
        echo "  ✓ Existing image removed"
    fi
    echo ""
fi

# Check if Docker image exists
IMAGE_EXISTS=$($DOCKER_CMD images tile-generator:latest -q 2>/dev/null || true)
if [ -z "$IMAGE_EXISTS" ]; then
    echo -e "${YELLOW}Building Docker image${NC}"
    echo ""
    echo "Steps that will be executed:"
    echo "  1. Pull Ubuntu 24.04 base image"
    echo "  2. Install Python 3 and development tools"
    echo "  3. Install system dependencies"
    echo ""
    echo "This may take a few minutes. Downloading and installing packages..."
    echo ""
    
    $DOCKER_CMD build --progress=plain -t tile-generator:latest -f /tmp/Dockerfile.tiles .
    
    echo ""
    echo -e "${GREEN}✓ Docker image created successfully${NC}"
    
    # Show image size
    IMAGE_SIZE=$($DOCKER_CMD images tile-generator:latest --format "{{.Size}}" 2>/dev/null || echo "unknown")
    echo "  Image size: $IMAGE_SIZE"
else
    echo -e "${GREEN}✓ Docker image already exists${NC}"
    
    # Show existing image size
    IMAGE_SIZE=$($DOCKER_CMD images tile-generator:latest --format "{{.Size}}" 2>/dev/null || echo "unknown")
    echo "  Image size: $IMAGE_SIZE"
fi

rm -f /tmp/Dockerfile.tiles

echo ""

# Check if gol is installed locally
GOL_PATH=""
if command -v gol &> /dev/null; then
    GOL_PATH=$(which gol)
    echo -e "${GREEN}Using local gol:${NC} $GOL_PATH"
else
    echo -e "${RED}✗ gol CLI not found in system${NC}"
    echo "Please install gol from: https://www.geodesk.com/download/"
    exit 1
fi

echo ""
echo "🚀 Starting tile generation..."
echo ""

# Run in Docker
$DOCKER_CMD run --rm \
    -v "$INPUT_ABS:/input/$INPUT_NAME" \
    -v "$CONFIG_ABS:/config/$CONFIG_NAME" \
    -v "$GOL_PATH:/gol:ro" \
    -v "$OUTPUT_ABS:/output" \
    -v "$(pwd)/tile_generator.py:/work/tile_generator.py" \
    tile-generator:latest bash -c "
        set -e
        
        echo '✓ Python dependencies already installed in image'
        echo ''
        
        # Step 1: Convert PBF to GOL
        echo '🔄 Step 1/2: Converting PBF to GOL...'
        echo \"  PBF: /input/$INPUT_NAME\"
        echo \"  GOL: input.gol\"
        echo ''
        echo 'Building GOL file...'
        
        # Start gol build in background and monitor progress (suppress all output)
        /gol build input.gol /input/$INPUT_NAME >/dev/null 2>/dev/null &
        GOL_PID=\$!
        
        # Monitor GOL file size while building
        while kill -0 \$GOL_PID 2>/dev/null; do
            if [ -f input.gol ]; then
                SIZE=\$(du -h input.gol 2>/dev/null | cut -f1 || echo \"0B\")
                echo -ne \"\\r  GOL size: \${SIZE} (building...)\"
            else
                echo -ne \"\\r  GOL size: 0B (building...)\"
            fi
            sleep 1
        done
        
        # Wait for gol to finish
        wait \$GOL_PID
        GOL_EXIT=\$?
        
        echo ''
        
        if [ \$GOL_EXIT -ne 0 ]; then
            echo ''
            echo '✗ Error converting PBF to GOL'
            echo ''
            echo 'Checking gol:'
            /gol --version || echo 'gol not working'
            exit 1
        fi
        
        if [ ! -f input.gol ]; then
            echo '✗ GOL file not created'
            echo 'Files in /work:'
            ls -lh /work/
            exit 1
        fi
        
        GOL_SIZE=\$(du -h input.gol | cut -f1)
        GOL_SIZE_KB=\$(du -k input.gol | cut -f1)
        PBF_SIZE_KB=\$(du -k /input/$INPUT_NAME | cut -f1)
        COMPRESSION=\$(awk \"BEGIN {printf \\\"%.1f\\\", (\$GOL_SIZE_KB/\$PBF_SIZE_KB)*100}\" 2>/dev/null || echo \"?\")
        echo ''
        echo '✓ GOL created successfully'
        echo \"  GOL size: \$GOL_SIZE (\${COMPRESSION}% of PBF size)\"
        echo ''
        
        # Step 2: Generate tiles from GOL
        echo '🗺️  Step 2/2: Generating tiles from GOL...'
        echo ''
        echo '  GOL: input.gol'
        echo \"  Config: /config/$CONFIG_NAME\"
        echo \"  Output: /output\"
        echo \"  Options: $ZOOM_ARGS\"
        echo ''
        
        python3 /work/tile_generator.py \
            input.gol \
            /output \
            /config/$CONFIG_NAME \
            $ZOOM_ARGS || {
            echo ''
            echo 'Checking if config file exists:'
            ls -lh /config/$CONFIG_NAME || echo 'Config not found'
            echo ''
            echo 'Checking /config directory:'
            ls -lh /config/ || echo '/config not found'
            exit 1
        }
        
        echo ''
        
        # Show results
        if [ -d \"/output\" ]; then
            TILE_COUNT=\$(find \"/output\" -name '*.bin' 2>/dev/null | wc -l)
            TOTAL_SIZE=\$(du -sh \"/output\" 2>/dev/null | cut -f1)
            echo '✓ Generation completed'
            echo \"  Tiles generated: \$TILE_COUNT\"
            echo \"  Total size: \$TOTAL_SIZE\"
        fi
    "

# Cleanup Docker resources
# Default behavior: KEEP the image for reuse
# Only clean if --clean-docker flag is set
if [ "$FORCE_CLEAN" = true ]; then
    echo ""
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    
    # Remove the Docker image created for this run
    if [ -n "$IMAGE_EXISTS" ]; then
        echo "  Removing Docker image: tile-generator:latest"
        $DOCKER_CMD rmi tile-generator:latest > /dev/null 2>&1
    fi
    
    # Remove any dangling images
    DANGLING_IMAGES=$($DOCKER_CMD images -f "dangling=true" -q 2>/dev/null | wc -l)
    if [ "$DANGLING_IMAGES" -gt 0 ]; then
        echo "  Removing dangling images..."
        $DOCKER_CMD rmi $($DOCKER_CMD images -f "dangling=true" -q) > /dev/null 2>&1
    fi
    
    echo -e "${GREEN}✓ Docker cleanup completed${NC}"
else
    # Default behavior: Keep the image for future runs
    echo ""
    echo -e "${GREEN}✓ Docker image kept for reuse (use --clean-docker to rebuild)${NC}"
    IMAGE_SIZE=$($DOCKER_CMD images tile-generator:latest --format "{{.Size}}" 2>/dev/null || echo "unknown")
    echo "  Image size: $IMAGE_SIZE"
fi
echo ""

echo -e "${GREEN}✓ Tiles generated in: $OUTPUT_DIR${NC}"
