#!/bin/bash
# Runs once when a fresh vanKonijnenburg-analyzer pod boots (set as RUNPOD_STARTUP_CMD / dockerArgs
# in lib/runpod.js's createPod()). Assumes a persistent network volume mounted at /workspace so the
# repo clone + pip installs only need to happen once ever, not on every cold start.
set -e
cd /workspace
if [ ! -d hallucheck ]; then
  git clone https://github.com/durand2511/hallucheck-scripts.git hallucheck
fi
cd hallucheck
pip install -q fastapi uvicorn[standard] transformers peft bitsandbytes accelerate torch datasets
# The adapter itself (models/dpo_gemma31b_grounding-adapter_v2/) is gitignored and NOT in this repo --
# it must already exist at /workspace/dpo_gemma31b_grounding-adapter_v2 on the network volume before
# this script runs (one-time manual seed step, done once per volume, not per pod boot).
python model_server.py
