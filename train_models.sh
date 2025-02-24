#!/usr/bin/zsh

for model in m_unet a_unet msaf_unet; do

    python train.py --model $model 

done

