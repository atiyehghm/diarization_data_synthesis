#!/bin/bash

# OPENRir Downloader
# Usage: ./download_openrir.sh [output_dir]

set -e

OUTPUT_DIR=${1:-"./data/OpenRIR"}

OPENRIR_URL="https://www.openslr.org/resources/28/rirs_noises.zip"

mkdir -p "${OUTPUT_DIR}"

echo "Starting download of OpenRIR dataset..."

wget -c "${OPENRIR_URL}" -O "${OUTPUT_DIR}/openrir.zip"

echo "Download completed!"

echo "Unzipping dataset..."
unzip "${OUTPUT_DIR}/openrir.zip" -d "${DEST_DIR}"

rm "${OUTPUT_DIR}/openrir.zip"