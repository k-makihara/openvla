#!/bin/bash

torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py   --model.type "prism-dinosiglip-224px-controlled+7b"   --model.model_id "minimum-pgvlm-cvsjpn-v5"   --model.vision_backbone_id "dinosiglip-vit-so-224px"   --model.image_resize_strategy "letterbox"   --model.llm_backbone_id "llama3.2-1b"

torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain_pref.py   --model.type "prism-dinosiglip-224px-controlled+7b"   --model.model_id "minimum-pgvlm-cvsjpn-pref-v3"   --model.vision_backbone_id "dinosiglip-vit-so-224px"   --model.image_resize_strategy "letterbox"   --model.llm_backbone_id "llama3.2-1b"