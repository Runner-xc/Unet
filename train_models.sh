#!/bin/bash

for model in unet DL_unet SED_unet; do

    python train.py --lr 8e-4 --wd 1e-6 --model $model 

done

