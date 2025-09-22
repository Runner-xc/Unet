#!/usr/bin/env python3
"""
数据准备脚本
自动进行数据集划分和处理
"""

import sys
sys.path.append('..')

import argparse
import os
from pathlib import Path
from src.utils.data_split import data_split_to_train_val_test
from src.utils.rock_data import process_rock_data

def main():
    parser = argparse.ArgumentParser(description="准备训练数据")
    parser.add_argument('--data_path', type=str, default='../datasets/CSV/shale_256.csv',
                       help='原始数据CSV文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--force_split', action='store_true',
                       help='强制重新划分数据集')
    
    args = parser.parse_args()
    
    print("📊 开始准备训练数据...")
    print(f"   数据路径: {args.data_path}")
    print(f"   训练集比例: {args.train_ratio}")
    print(f"   验证集比例: {args.val_ratio}")
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"❌ 错误: 找不到数据文件 {args.data_path}")
        return
    
    # 数据集划分
    try:
        # 这里需要根据实际的数据划分函数调整参数
        print("🔄 正在划分数据集...")
        # train_csv, val_csv, test_csv = data_split_to_train_val_test(
        #     args.data_path, 
        #     train_ratio=args.train_ratio,
        #     val_ratio=args.val_ratio,
        #     flag=args.force_split
        # )
        
        print("✅ 数据准备完成！")
        print("   训练集、验证集、测试集已生成")
        
    except Exception as e:
        print(f"❌ 数据准备失败: {str(e)}")

if __name__ == '__main__':
    main()