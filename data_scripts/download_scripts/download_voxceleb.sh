#!/bin/bash

# VoxCeleb 1&2 Downloader
# Usage: ./download_voxceleb.sh <output_dir>

set -e

OUTPUT_DIR=${1:-"./data/voxceleb"}
mkdir -p "$OUTPUT_DIR"


VOX1_URL="http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa"
VOX2_URL="http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2_aac.zip"

echo "Downloading VoxCeleb1..."
wget -q --show-progress --continue -P "$OUTPUT_DIR" "$VOX1_URL"

echo "Downloading VoxCeleb2..."
wget -q --show-progress --continue -P "$OUTPUT_DIR" "$VOX2_URL"

echo "Unpacking archives..."
7z x "$OUTPUT_DIR/vox1_dev_wav_partaa" -o"$OUTPUT_DIR/vox1"
7z x "$OUTPUT_DIR/vox2_aac.zip" -o"$OUTPUT_DIR/vox2"

echo "Cleaning up..."
rm "$OUTPUT_DIR"/vox1_dev_wav_partaa "$OUTPUT_DIR"/vox2_aac.zip

echo "VoxCeleb download complete!"