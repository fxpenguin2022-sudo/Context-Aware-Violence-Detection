#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# This example keeps the release surface small and demonstrates the paper's
# main joint model. It assumes the branch checkpoints have already been
# prepared and placed under ./checkpoints/ or provided via environment vars.
POSE_BRANCH_CKPT="${POSE_BRANCH_CKPT:-./checkpoints/pose_branch_best_acc.pt}"
SCENE_DECOUPLING_CKPT="${SCENE_DECOUPLING_CKPT:-./checkpoints/scene_decoupling_best_acc.pt}"

python joint_cross_attention/scripts/preprocess_index.py \
  --config \
  joint_cross_attention/configs/base.yaml \
  joint_cross_attention/configs/data/rwf2000_video_pose_joint.yaml

python joint_cross_attention/scripts/train.py \
  --config \
  joint_cross_attention/configs/base.yaml \
  joint_cross_attention/configs/data/rwf2000_video_pose_joint.yaml \
  joint_cross_attention/configs/model/joint_sg_aca_amcf.yaml \
  joint_cross_attention/configs/train/default.yaml \
  joint_cross_attention/configs/exp/rwf2000_joint_model.yaml \
  --override \
  model.pose_branch.checkpoint="${POSE_BRANCH_CKPT}" \
  model.context_branch.checkpoint="${SCENE_DECOUPLING_CKPT}"
