#!/bin/bash
set -e

COLLECT_PATH=./uploads/

# Create target folder if doesn't exist:
if [[ -d "${COLLECT_PATH}" ]]; then
    mkdir -p "${COLLECT_PATH}"
fi

# Copy
# scp -r bidmachine:/var/www/labelar-backend/uploads/* "${COLLECT_PATH}"
# RSYNC is faster than scp, so use this instead:
rsync -azvh bidmachine:/var/www/labelar-backend/uploads/ "${COLLECT_PATH}"