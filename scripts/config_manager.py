#!/usr/bin/env python3
"""
简化的配置管理
提供快速配置模板
"""

import os
import argparse
from pathlib import Path

def create_config_template(model_name, output_path):
    """创建配置文件模板"""
    
    template = f"""# {model_name.upper()} 配置文件
# 基于YAML的训练配置

MODEL:
  NAME: '{model_name}'
  RESUME: None 
  DROPOUT_P: 0.2

DATA:
  DATA_PATH: './datasets/CSV/shale_256.csv'
  CLASS_NAMES: ['OM', 'OP', 'IOP']
  IMG_SIZE: 256
  BATCH_SIZE: 8
  VALID_BATCH_SIZE: 8
  TRAIN_RATIO: 0.7
  VAL_RATIO: 0.2
  SPLIT_FLAG: False
  NUM_SMALL_DATA: None

TRAIN:
  EPOCHS: 200
  START_EPOCH: 0
  EVAL_INTERVAL: 1
  AMP: True
  EARLY_STOP_PATIENCE: 30

  OPTIMIZER: 
    NAME: 'AdamW'
    BASE_LR: 5e-4
    WEIGHT_DECAY: 0.0
    ADAMW_BETAS: [0.95, 0.999]
    SGD_MOMENTUM: 0.9

  SCHEDULER:
    NAME: 'CosineAnnealingLR'
    WARMUP_EPOCHS: 5
    COSINE_T_MAX: 60
    COSINE_ETA_MIN: 1e-8
    PLATEAU_FACTOR: 0.1
    PLATEAU_PATIENCE: 5

  LOSS:
    NAME: 'DiceLoss'
    ELN_LOSS: False
    L1_LAMBDA: 1e-4
    L2_LAMBDA: 1e-4

SYSTEM:
  DEVICE: 'cuda:0'
  NUM_WORKERS: 8
  PIN_MEMORY: True

LOG:
  USE_TENSORBOARD: True
  USE_SWANLAB: True
  SAVE_FLAG: True
  MDF_PARAMS: False

OUTPUT: './training_results'
TAG: 'default'
SEED: 42
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"✅ 配置文件已创建: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="配置文件管理工具")
    parser.add_argument('--create', type=str, help='创建指定模型的配置文件')
    parser.add_argument('--output', type=str, help='输出配置文件路径')
    parser.add_argument('--list', action='store_true', help='列出所有可用配置')
    
    args = parser.parse_args()
    
    configs_dir = Path('configs')
    
    if args.list:
        print("📋 可用的配置文件:")
        if configs_dir.exists():
            for config_file in configs_dir.glob('*.yaml'):
                print(f"   - {config_file.name}")
        else:
            print("   暂无配置文件")
        return
    
    if args.create:
        if not args.output:
            args.output = f"configs/{args.create}_custom.yaml"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        create_config_template(args.create, args.output)
        
        print(f"📝 使用方法:")
        print(f"   python train.py --cfg {args.output}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()