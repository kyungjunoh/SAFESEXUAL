#!/bin/bash
# IFGR — Image-Feedback Gated Regeneration (main method).
# Pass 1: SAFREE generation (word-based P_c, no corpus -> leak-free).
# Gate:   independent CLIP zero-shot NSFW detector (separate from the NudeNet evaluator).
# Pass 2: regenerate flagged images (same seed) with strong cross-attention suppression.
#
# Usage:  bash scripts/run_ifgr.sh
# Eval on P4D (datasets/nudity.csv); ASR is measured by NudeNet.

SD_MODEL_ID=v1-4
MODEL_ID="CompVis/stable-diffusion-v1-4"

python generate_safree.py \
    --config ./configs/sd_config.json \
    --data ./datasets/nudity.csv \
    --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
    --category nudity \
    --num-samples 1 \
    --erase-id std \
    --model_id $MODEL_ID \
    --nudity_thr 0.6 \
    --device cuda:0 \
    --save-dir ./results/gen_IFGR_SD${SD_MODEL_ID}_p4d/ \
    --safree -svf -lra \
    --latent_feedback --lf_lambda 1.0 --lf_gate clip --lf_gate_thr 0.8

# Optional (appendix, stronger): data-driven P_c from a held-out NSFW corpus.
# Add:  --pc_corpus ./datasets/i2p_nudity_heldout2.csv --pc_nudity_thr 0 --pc_rank 60
