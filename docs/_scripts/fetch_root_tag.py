#!/usr/bin/env python3
"""Download and extract ROOT Doxygen tag file."""

# ruff: noqa: T201

from __future__ import annotations

import gzip
import os
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

# Get ROOT version from environment variable or use default
ROOT_VERSION = os.environ.get("ROOT_DOXYGEN_VERSION", "v636")

# Paths
DOCS_DIR = Path(__file__).parent.parent
TAG_FILE = DOCS_DIR / "ROOT.tag"
TAG_URL = f"https://root.cern/doc/{ROOT_VERSION}/ROOT.tag.gz"


def main():
    """Download and extract ROOT.tag if it doesn't exist."""
    # Skip if already exists (idempotent)
    if TAG_FILE.exists():
        print(f"✓ {TAG_FILE} already exists, skipping download")
        return 0

    # Validate URL uses HTTPS for security
    if not TAG_URL.startswith("https://"):
        print(f"✗ URL must start with 'https://': {TAG_URL}", file=sys.stderr)
        return 1

    print(f"Downloading {TAG_URL}...")
    try:
        with urlopen(TAG_URL) as response:
            if response.status != 200:
                print(f"✗ Failed to download: HTTP {response.status}", file=sys.stderr)
                return 1

            # Decompress gzip directly to file
            with (
                gzip.open(response, "rb") as gz_file,
                TAG_FILE.open("wb") as out_file,
            ):
                shutil.copyfileobj(gz_file, out_file)

        print(f"✓ Downloaded and extracted to {TAG_FILE}")
        return 0

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
