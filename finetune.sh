#!/bin/bash

ln -s /workspace/metl/data/
ln -s /workspace/metl/pretrained_models/

python /workspace/metl/code/train_target_model.py @/workspace/metl/args/finetune_avgfp_local.txt --enable_progress_bar false --enable_simple_progress_messages --max_epochs 50 --unfreeze_backbone_at_epoch 25
