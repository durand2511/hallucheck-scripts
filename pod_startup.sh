#!/bin/bash
# Runs once when a fresh vanKonijnenburg-analyzer pod boots (set as RUNPOD_STARTUP_CMD / dockerArgs
# in lib/runpod.js's createPod()). Assumes a persistent network volume mounted at /workspace so the
# repo clone + pip installs only need to happen once ever, not on every cold start.
#
# dockerArgs REPLACES the container's own entrypoint, so if this script's process exits (crash or
# otherwise), RunPod restarts the whole container -- taking sshd down with it every time, which
# made every earlier crash impossible to actually inspect (SSH just went unreachable). Everything
# below is tee'd to a persistent log on the volume, and a crash drops into `sleep infinity` instead
# of letting the container exit, so the pod (and SSH) stays alive for inspection until the app's
# own health-check timeout terminates it.
{
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
# Pinned combination confirmed working via a live pod (2026-08-19):
# - google/gemma-4-31B-it's config declares model_type "gemma4", which only got added to
#   transformers in a release that also hard-requires PyTorch >= 2.5 -- the runpod/pytorch base
#   image ships torch 2.4.1, so an old-enough transformers to run on 2.4.1 doesn't recognize
#   "gemma4" at all (KeyError: 'gemma4'), and a new-enough transformers to recognize it silently
#   disables its own PyTorch integration on 2.4.1 (NameError: name 'torch' is not defined, deep in
#   transformers.integrations.tensor_parallel). The only way through is upgrading torch itself.
# - Upgrading only torch breaks the base image's preinstalled torchvision/torchaudio (built against
#   2.4.1) with "RuntimeError: operator torchvision::nms does not exist" the moment transformers
#   pulls in an unrelated image-processing module -- torchvision has to move in lockstep with torch.
pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -q fastapi uvicorn[standard] -U transformers peft accelerate bitsandbytes datasets
# The adapter itself (models/dpo_gemma31b_grounding-adapter_v2/) is gitignored and NOT in this repo --
# it must already exist at /workspace/dpo_gemma31b_grounding-adapter_v2 on the network volume before
# this script runs (one-time manual seed step, done once per volume, not per pod boot).
python model_server.py
} > >(tee -a /workspace/boot.log) 2>&1
echo "=== container process exiting, dropping into sleep so the pod stays inspectable ===" >> /workspace/boot.log
sleep infinity
