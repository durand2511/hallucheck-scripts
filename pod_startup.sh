#!/bin/bash
# Runs once when a fresh vanKonijnenburg-analyzer pod boots (set as RUNPOD_STARTUP_CMD / dockerArgs
# in lib/runpod.js's createPod()). Assumes a persistent network volume mounted at /workspace so the
# repo clone + pip installs only need to happen once ever, not on every cold start.
set -e
cd /workspace
# hallucheck-scripts is a private repo, so this needs a token (GITHUB_PAT, set as an env var on
# the pod itself by lib/runpod.js's createPod() -- fine-grained, read-only, scoped to this one repo).
if [ ! -d hallucheck ]; then
  git clone "https://${GITHUB_PAT}@github.com/durand2511/hallucheck-scripts.git" hallucheck
else
  (cd hallucheck && git pull) || true
fi
cd hallucheck
# Pinned: unpinned "latest" transformers now requires PyTorch >= 2.5, but the runpod/pytorch base
# image ships 2.4.1 -- installing latest transformers silently disables its own PyTorch integration
# and crashes deep inside transformers.integrations.tensor_parallel with "NameError: name 'torch'
# is not defined" the moment peft imports it. Confirmed working combination via a live pod (2026-08-19).
pip install -q fastapi uvicorn[standard] "transformers==4.46.3" "peft==0.13.2" "accelerate==1.0.1" bitsandbytes datasets
# The adapter itself (models/dpo_gemma31b_grounding-adapter_v2/) is gitignored and NOT in this repo --
# it must already exist at /workspace/dpo_gemma31b_grounding-adapter_v2 on the network volume before
# this script runs (one-time manual seed step, done once per volume, not per pod boot).
python model_server.py
