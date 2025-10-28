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

def get_image_files(directory):
    """获取指定目录下的所有.nii.gz文件"""
    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]
    return sorted(files)

def create_dataset_split():
    """
    生成数据集划分文件
    - splitdataset.pkl: 5折交叉验证的训练/验证集划分
    - testdataset.pkl: 测试集文件列表（未标注数据，用于半监督学习）
    """
    
    # 定义目录路径 - 相对于configs目录，需要上一级
    train_image_dir = '../dataset/processdata2D/image'
    train_label_dir = '../dataset/processdata2D/mask'
    test_image_dir = '../dataset/imagesTs'  # 半监督学习的未标注数据
    
    print("Reading training files...")
    # 获取所有训练图像文件
    train_images = get_image_files(train_image_dir)
    
    # 检查对应的标签文件是否存在
    train_files = []
    for img_file in train_images:
        label_file = os.path.join(train_label_dir, img_file)
        if os.path.exists(label_file):
            train_files.append(img_file)
        else:
            print(f"Warning: Corresponding label file not found: {img_file}")
    
    print(f"Found {len(train_files)} training samples")
    
    # 获取测试集文件（用于半监督学习的未标注数据）
    test_files = get_image_files(test_image_dir)
    print(f"Found {len(test_files)} test samples (for semi-supervised learning)")
    
    # 5折交叉验证划分
    n_splits = 5
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
    
    # 保存划分结果
    print("\nSaving splitdataset.pkl...")
    with open('splitdataset.pkl', 'wb') as f:
        pk.dump(dataset_split, f)
    print("splitdataset.pkl saved")
    
    # 保存测试集文件列表（用于半监督学习）
    print("\nSaving testdataset.pkl...")
    with open('testdataset.pkl', 'wb') as f:
        pk.dump(test_files, f)
    print(f"testdataset.pkl saved (containing {len(test_files)} unlabeled samples)")
    
    print("\nDataset split completed!")
    print("=" * 50)
    print("Generated files:")
    print("- splitdataset.pkl: 5-fold cross-validation train/val splits")
    print("- testdataset.pkl: Test set file list (for semi-supervised learning)")
    print("=" * 50)

if __name__ == '__main__':
    create_dataset_split()

