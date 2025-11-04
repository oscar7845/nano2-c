#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./build/nano2 --config ./configs/nano2.json || true
