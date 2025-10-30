#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate dataset split files (splitdataset.pkl and testdataset.pkl) for SSHSNet
根据训练集和测试集数据生成5折交叉验证的数据划分
"""

import os
import pickle as pk
from sklearn.model_selection import KFold
import numpy as np
import argparse

def get_image_files(directory):
    """获取指定目录下的所有.nii.gz文件"""
    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]
    return sorted(files)

def create_dataset_split(train_image_dir: str, train_label_dir: str, test_image_dir: str):
    """
    生成数据集划分文件
    - splitdataset.pkl: 5折交叉验证的训练/验证集划分
    - testdataset.pkl: 测试集文件列表（未标注数据，用于半监督学习）
    """
    
    # 规范化为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def to_abs(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(script_dir, p))

    train_image_dir = to_abs(train_image_dir)
    train_label_dir = to_abs(train_label_dir)
    test_image_dir = to_abs(test_image_dir)  # 半监督学习的未标注数据

    print("Directories:")
    print(f"  train_image_dir: {train_image_dir}")
    print(f"  train_label_dir: {train_label_dir}")
    print(f"  test_image_dir : {test_image_dir}")

    # 基本校验
    if not os.path.isdir(train_image_dir):
        raise FileNotFoundError(f"Training images directory not found: {train_image_dir}")
    if not os.path.isdir(train_label_dir):
        raise FileNotFoundError(f"Training labels directory not found: {train_label_dir}")
    if not os.path.isdir(test_image_dir):
        print(f"Warning: Test images directory not found: {test_image_dir}. Proceeding with empty test set.")
    
    print("Reading training files...")
    # 获取所有训练图像文件
    train_images = get_image_files(train_image_dir)
    
    # 检查对应的标签文件是否存在（兼容多种命名：同名；mask_ 前缀；Case→case + 前缀）
    def match_label(filename: str) -> bool:
        # 1) same name
        direct = os.path.join(train_label_dir, filename)
        if os.path.exists(direct):
            return True
        # 2) mask_ + same name
        prefixed = os.path.join(train_label_dir, f"mask_{filename}")
        if os.path.exists(prefixed):
            return True
        # 3) mask_ + Case→case
        replaced = os.path.join(train_label_dir, f"mask_{filename.replace('Case', 'case')}")
        if os.path.exists(replaced):
            return True
        # 4) Case→case (no prefix)
        lower_case = os.path.join(train_label_dir, filename.replace('Case', 'case'))
        if os.path.exists(lower_case):
            return True
        return False

    train_files = []
    for img_file in train_images:
        if match_label(img_file):
            train_files.append(img_file)
        else:
            print(f"Warning: Corresponding label file not found: {img_file}")
    
    print(f"Found {len(train_files)} training samples")
    
    # 获取测试集文件（用于半监督学习的未标注数据）
    test_files = get_image_files(test_image_dir)
    print(f"Found {len(test_files)} test samples (for semi-supervised learning)")
    
    # 交叉验证划分（样本过少时自动调整折数）
    if len(train_files) < 2:
        raise ValueError(f"Not enough labeled samples for CV: {len(train_files)} found. Need at least 2.")
    n_splits = min(5, len(train_files))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2021)
    
    # 生成5折交叉验证的划分
    dataset_split = []
    for train_idx, val_idx in kf.split(train_files):
        train_split = [train_files[i] for i in train_idx]
        val_split = [train_files[i] for i in val_idx]
        dataset_split.append([train_split, val_split])
        print(f"\nFold {len(dataset_split)}:")
        print(f"  Training set: {len(train_split)} samples")
        print(f"  Validation set: {len(val_split)} samples")
    
    # 保存划分结果到脚本所在目录（configs/）
    print("\nSaving splitdataset.pkl...")
    out_split = os.path.join(script_dir, 'splitdataset.pkl')
    with open(out_split, 'wb') as f:
        pk.dump(dataset_split, f)
    print(f"splitdataset.pkl saved -> {out_split}")
    
    # 保存测试集文件列表（用于半监督学习）
    print("\nSaving testdataset.pkl...")
    out_test = os.path.join(script_dir, 'testdataset.pkl')
    with open(out_test, 'wb') as f:
        pk.dump(test_files, f)
    print(f"testdataset.pkl saved -> {out_test} (containing {len(test_files)} unlabeled samples)")
    
    print("\nDataset split completed!")
    print("=" * 50)
    print("Generated files:")
    print("- splitdataset.pkl: 5-fold cross-validation train/val splits")
    print("- testdataset.pkl: Test set file list (for semi-supervised learning)")
    print("=" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate splitdataset.pkl and testdataset.pkl')
    parser.add_argument('--train_image_dir', type=str, default='/root/SSHSNet/dataset/MR', help='Path to training images (NIfTI)')
    parser.add_argument('--train_label_dir', type=str, default='/root/SSHSNet/dataset/Mask', help='Path to training labels (NIfTI)')
    parser.add_argument('--test_image_dir', type=str, default='/root/SSHSNet/dataset/imagesTs', help='Path to unlabeled test images (NIfTI)')
    args = parser.parse_args()

    create_dataset_split(
        train_image_dir=args.train_image_dir,
        train_label_dir=args.train_label_dir,
        test_image_dir=args.test_image_dir,
    )

