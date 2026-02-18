#!/bin/bash

# MUSAN and MS-SNSD Downloader
# Usage: ./download_musan.sh [output_dir]

set -e

OUTPUT_DIR=${1:-"./data/musan"}
mkdir -p "$OUTPUT_DIR"

# MUSAN
echo "Downloading MUSAN..."
git clone https://github.com/facebookresearch/musan.git "$OUTPUT_DIR/musan"

# MS-SNSD
echo "Downloading MS-SNSD..."
wget -q --show-progress --continue -P "$OUTPUT_DIR" \
    "https://github.com/microsoft/MS-SNSD/archive/refs/heads/master.zip"

unzip "$OUTPUT_DIR/master.zip" -d "$OUTPUT_DIR/ms-snsd"
rm "$OUTPUT_DIR/master.zip"

python3 noise_index.py