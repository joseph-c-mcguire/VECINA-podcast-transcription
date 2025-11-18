#!/bin/bash
# VECINA Podcast Transcription Pipeline
# This script uploads audio files to Modal, runs transcription, and downloads results

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default directories
AUDIO_DIR="${1:-_data/podcasts}"
OUTPUT_DIR="${2:-_data/_output}"

echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  ██╗   ██╗███████╗ ██████╗██╗███╗   ██╗ █████╗                ║
║  ██║   ██║██╔════╝██╔════╝██║████╗  ██║██╔══██╗               ║
║  ██║   ██║█████╗  ██║     ██║██╔██╗ ██║███████║               ║
║  ╚██╗ ██╔╝██╔══╝  ██║     ██║██║╚██╗██║██╔══██║               ║
║   ╚████╔╝ ███████╗╚██████╗██║██║ ╚████║██║  ██║               ║
║    ╚═══╝  ╚══════╝ ╚═════╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝               ║
║                                                                ║
║           Podcast Transcription Pipeline                      ║
║           RIOS Institute - Community Involvement              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

# Step 1: Upload audio files
echo -e "${YELLOW}📤 Step 1/3: Uploading audio files to Modal...${NC}"
echo -e "${BLUE}   Source: ${AUDIO_DIR}${NC}"
echo ""
modal run scripts/upload_to_modal.py --audio-dir "$AUDIO_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Upload failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Upload complete!${NC}"
echo ""

# Step 2: Run transcription
echo -e "${YELLOW}🎙️  Step 2/3: Running batch transcription...${NC}"
echo -e "${BLUE}   This may take a while depending on file sizes and chunk settings${NC}"
echo ""
modal run vecina_transcriber/modal_entrypoint.py::transcribe_all_modal

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Transcription failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Transcription complete!${NC}"
echo ""

# Step 3: Download results
echo -e "${YELLOW}📥 Step 3/3: Downloading transcripts...${NC}"
echo -e "${BLUE}   Destination: ${OUTPUT_DIR}${NC}"
echo ""
modal run scripts/download_from_modal.py --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Download failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   ✅ PIPELINE COMPLETE!                               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "   Audio source: ${AUDIO_DIR}"
echo -e "   Transcripts saved to: ${OUTPUT_DIR}"
echo ""
echo -e "${BLUE}💡 Tip: Check your transcripts in ${OUTPUT_DIR}${NC}"
