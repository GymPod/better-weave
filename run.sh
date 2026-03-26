#!/bin/bash
# Run Better Weave locally (recommended - has access to internal W&B and AWS Bedrock)
cd "$(dirname "$0")"
echo "Starting Better Weave on http://localhost:8421"
echo "Press Ctrl+C to stop"
python -m uvicorn app:app --host 0.0.0.0 --port 8421 --reload
