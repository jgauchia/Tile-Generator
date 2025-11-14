#!/bin/bash

# Script to generate tiles from PBF using Docker
# Converts PBF -> GOL -> Tiles automatically
# Optimized version with minimal dependencies (NO SHAPELY)

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
echo "║   PBF → GOL → Tiles Generator          ║"
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
    echo "  --zoom N-M          Zoom range (e.g., 6-17)"
    echo "  --max-file-size KB  Maximum tile file size in KB"
    echo ""
    echo "Example:"
    echo "  $0 map.osm.pbf tiles features.json --zoom 6-17"
    echo "  $0 map.osm.pbf tiles features.json --zoom 12 --max-file-size 512"
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

# Create output directory
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

# Create lightweight Dockerfile (NO SHAPELY, NO OSMIUM)
cat > /tmp/Dockerfile.tiles.light << 'EOF'
FROM python:3.11-slim

# Install only essential dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install minimal Python dependencies (NO SHAPELY)
# Total size: ~20MB vs ~200MB with shapely
RUN pip install --no-cache-dir \
    ijson==3.2.3 \
    tqdm==4.66.1 \
    psutil==5.9.8

WORKDIR /work

# Display installed packages for verification
RUN pip list && \
    echo "Docker image ready - lightweight version (no shapely)"
EOF

# Handle --clean-docker option
if [ "$FORCE_CLEAN" = true ]; then
    echo -e "${YELLOW}--clean-docker flag: rebuilding Docker image${NC}"
    IMAGE_EXISTS=$($DOCKER_CMD images tile-generator:light -q 2>/dev/null || true)
    if [ -n "$IMAGE_EXISTS" ]; then
        echo "  Removing existing image..."
        $DOCKER_CMD rmi tile-generator:light > /dev/null 2>&1 || true
        echo "  ✓ Existing image removed"
    fi
    echo ""
fi

# Check if Docker image exists
IMAGE_EXISTS=$($DOCKER_CMD images tile-generator:light -q 2>/dev/null || true)
if [ -z "$IMAGE_EXISTS" ]; then
    echo -e "${YELLOW}Building lightweight Docker image${NC}"
    echo ""
    
    $DOCKER_CMD build --progress=plain -t tile-generator:light -f /tmp/Dockerfile.tiles.light .
    
    echo ""
    echo -e "${GREEN}✓ Lightweight Docker image created${NC}"
    
    # Show image size
    IMAGE_SIZE=$($DOCKER_CMD images tile-generator:light --format "{{.Size}}" 2>/dev/null || echo "unknown")
    echo "  Image size: $IMAGE_SIZE"
else
    echo -e "${GREEN}✓ Docker image already exists${NC}"
    IMAGE_SIZE=$($DOCKER_CMD images tile-generator:light --format "{{.Size}}" 2>/dev/null || echo "unknown")
    echo "  Image size: $IMAGE_SIZE"
fi

rm -f /tmp/Dockerfile.tiles.light

echo ""

# Check if gol is installed locally
GOL_PATH=""
if command -v gol &> /dev/null; then
    GOL_PATH=$(which gol)
    echo -e "${GREEN}Using local gol:${NC} $GOL_PATH"
else
    echo -e "${RED}✗ gol CLI not found${NC}"
    echo "Install from: https://www.geodesk.com/download/"
    exit 1
fi

echo ""
echo "🚀 Starting tile generation (lightweight mode)..."
echo ""

# Run in Docker
$DOCKER_CMD run --rm \
    -v "$INPUT_ABS:/input/$INPUT_NAME:ro" \
    -v "$CONFIG_ABS:/config/$CONFIG_NAME:ro" \
    -v "$GOL_PATH:/gol:ro" \
    -v "$OUTPUT_ABS:/output" \
    -v "$(pwd)/tile_generator.py:/work/tile_generator.py:ro" \
    tile-generator:light bash -c "
        set -e
        
        echo '✓ Lightweight Python environment ready'
        echo ''
        
        # Step 1: Convert PBF to GOL
        echo '🔄 Step 1/2: Converting PBF to GOL...'
        echo \"  PBF: /input/$INPUT_NAME\"
        echo \"  GOL: input.gol\"
        echo ''
        
        # Build GOL with progress monitoring
        /gol build input.gol /input/$INPUT_NAME >/dev/null 2>&1 &
        GOL_PID=\$!
        
        # Monitor progress
        while kill -0 \$GOL_PID 2>/dev/null; do
            if [ -f input.gol ]; then
                SIZE=\$(du -h input.gol 2>/dev/null | cut -f1 || echo \"0B\")
                echo -ne \"\\r  GOL size: \${SIZE} (building...)\"
            fi
            sleep 1
        done
        
        echo ''
        wait \$GOL_PID
        
        if [ ! -f input.gol ]; then
            echo '✗ GOL file not created'
            exit 1
        fi
        
        GOL_SIZE=\$(du -h input.gol | cut -f1)
        echo '✓ GOL created successfully'
        echo \"  Size: \$GOL_SIZE\"
        echo ''
        
        # Step 2: Generate tiles (NO SHAPELY - pure Python)
        echo '🗺️  Step 2/2: Generating tiles ...'
        echo ''
        
        python3 /work/tile_generator.py \
            input.gol \
            /output \
            /config/$CONFIG_NAME \
            $ZOOM_ARGS
        
        echo ''
        
        # Show results
        if [ -d \"/output\" ]; then
            echo '✓ Tiles generation completed'
        fi
    "

# Docker cleanup logic
if [ "$FORCE_CLEAN" = true ]; then
    echo ""
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    
    if [ -n "$IMAGE_EXISTS" ]; then
        echo "  Removing Docker image: tile-generator:light"
        $DOCKER_CMD rmi tile-generator:light > /dev/null 2>&1 || true
    fi
    
    # Clean dangling images
    DANGLING=$($DOCKER_CMD images -f "dangling=true" -q 2>/dev/null | wc -l)
    if [ "$DANGLING" -gt 0 ]; then
        echo "  Removing dangling images..."
        $DOCKER_CMD rmi $($DOCKER_CMD images -f "dangling=true" -q) > /dev/null 2>&1 || true
    fi
    
    echo -e "${GREEN}✓ Docker cleanup completed${NC}"
else
    echo ""
    echo -e "${GREEN}✓ Docker image kept for reuse${NC}"
    echo "  Use --clean-docker to rebuild"
    IMAGE_SIZE=$($DOCKER_CMD images tile-generator:light --format "{{.Size}}" 2>/dev/null || echo "unknown")
    echo "  Image size: $IMAGE_SIZE (lightweight)"
fi

echo ""
echo -e "${GREEN}✓ Tiles generated in: $OUTPUT_DIR${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo "  • Docker image kept for reuse"
echo "  • Use --clean-docker to rebuild"
