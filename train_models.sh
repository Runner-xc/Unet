#!/bin/bash

for wd in 1e-2 1e-3 1e-4 1e-6; do

    python train.py --lr 8e-4 --wd $wd --model DL_unet 

done

