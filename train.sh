#!/bin/bash
#
#for i in 0 1 2 3
#do
#    python train.py \
#        --device-ids 0,1 \
#        --batch-size 6 \
#        --fold $i \
#        --workers 12 \
#        --lr 0.0001 \
#        --n-epochs 10 \
#        --jaccard-weight 0.3 \
#        --model UNet \
#        --train_crop_height 1024 \
#        --train_crop_width 1280 \
#        --val_crop_height 1024 \
#        --val_crop_width 1280
#    python train.py \
#        --device-ids 0,1 \
#        --batch-size 6 \
#        --fold $i \
#        --workers 12 \
#        --lr 0.00001 \
#        --n-epochs 20 \
#        --jaccard-weight 0.3 \
#        --model UNet \
#        --train_crop_height 1024 \
#        --train_crop_width 1280 \
#        --val_crop_height 1024 \
#        --val_crop_width 1280
#done
#python train.py \
#    --lr 0.0001 \
#    --n-epochs 15

#python train.py \
#    --lr 0.0001 \
#    --n-epochs 30 \
#
#python train.py \
#    --lr 0.00008 \
#    --n-epochs 60 \
#
#python train.py \
#    --lr 0.00005 \
#    --n-epochs 90 \
#
#python train.py \
#    --lr 0.00005 \
#    --n-epochs 90 \
#
#python train.py \
#    --lr 0.00004 \
#    --n-epochs 120 \
#
#python train.py \
#    --lr 0.00003 \
#    --n-epochs 150 \
#
#python train.py \
#    --lr 0.00002 \
#    --n-epochs 181 \
#
#python train.py \
#    --lr 0.00001 \
#    --n-epochs 210 \
#
#python train.py \
#    --lr 0.000008 \
#    --n-epochs 250 \

#python train.py \
#    --lr 0.000007 \
#    --n-epochs 300 \
#
#python train.py \
#    --lr 0.000006 \
#    --n-epochs 400 \

python train.py \
    --lr 0.0001 \
    --batch-size 2 \
    --n-epochs 400 \
    --clip 5\
    --model_type 'transformer_fusion' \
    --model MultiFrameFusionUNet11_ViTViTv2 \
    --heads 24 \


#python train.py \
#    --lr 0.0001 \
#    --n-epochs 450 \
#    --model TeRAUNet \
#    --heads 8 \
