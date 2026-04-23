#!/bin/bash
set -e

# Get the directory where the script is located
VIZ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FRONTEND_DIR="$VIZ_DIR/frontend"
BACKEND_DIR="$VIZ_DIR/backend"

echo "Building StatPP Viz Release..."

# 1. Build Frontend
echo "--- Building Frontend ---"
cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi
npm run build

# 2. Build Backend (with embedded frontend)
echo "--- Building Backend ---"
cd "$BACKEND_DIR"
cargo build --release

# 3. Finalize
BINARY_NAME="backend"
# You might want to rename it to something more descriptive
# mv "$BACKEND_DIR/target/release/$BINARY_NAME" "$BACKEND_DIR/target/release/statpp-viz"

echo "--- Build Complete ---"
echo "Binary: $BACKEND_DIR/target/release/$BINARY_NAME"
