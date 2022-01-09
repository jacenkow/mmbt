#!/bin/bash

for i in {1..3}  # Three seeds.
do
  mmf_run \
    env.data_dir="datasets" \
    env.save_dir="training/save_chestbert_$i" \
    env.user_dir="chest_multimodal" training.tensorboard="True" training.seed="$i" \
    config="chest_multimodal/configs/experiments/chestbert/defaults.yaml" \
    model="chestbert" dataset="mimic_cxr"
  mmf_predict \
    env.data_dir="datasets" \
    env.save_dir="training/predict_chestbert_$i" \
    env.user_dir="chest_multimodal" \
    config="chest_multimodal/configs/experiments/chestbert/defaults.yaml" \
    model="chestbert" dataset="mimic_cxr" run_type="test" \
    checkpoint.resume_file="training/save_chestbert_$i/best.ckpt" \
    checkpoint.resume_pretrained=False \
    training.num_workers=0
done
