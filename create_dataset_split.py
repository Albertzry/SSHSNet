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
    
    # 定义目录路径
    train_image_dir = './dataset/imagesTr'
    train_label_dir = './dataset/labelsTr'
    test_image_dir = './dataset/imagesTs'
    
    print("正在读取训练集文件...")
    # 获取所有训练图像文件
    train_images = get_image_files(train_image_dir)
    
    # 检查对应的标签文件是否存在
    train_files = []
    for img_file in train_images:
        label_file = os.path.join(train_label_dir, img_file)
        if os.path.exists(label_file):
            train_files.append(img_file)
        else:
            print(f"警告: 未找到对应的标签文件 {img_file}")
    
    print(f"找到 {len(train_files)} 个训练样本")
    
    # 获取测试集文件（用于半监督学习的未标注数据）
    test_files = get_image_files(test_image_dir)
    print(f"找到 {len(test_files)} 个测试样本（用于半监督学习）")
    
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
        print(f"  训练集: {len(train_split)} 个样本")
        print(f"  验证集: {len(val_split)} 个样本")
    
    # 保存划分结果
    print("\n正在保存 splitdataset.pkl...")
    with open('splitdataset.pkl', 'wb') as f:
        pk.dump(dataset_split, f)
    print("splitdataset.pkl 已保存")
    
    # 保存测试集文件列表（用于半监督学习）
    print("\n正在保存 testdataset.pkl...")
    with open('testdataset.pkl', 'wb') as f:
        pk.dump(test_files, f)
    print(f"testdataset.pkl 已保存（包含 {len(test_files)} 个未标注样本）")
    
    print("\n数据集划分完成！")
    print("=" * 50)
    print("生成的文件:")
    print("- splitdataset.pkl: 5折交叉验证的训练/验证集划分")
    print("- testdataset.pkl: 测试集文件列表（用于半监督学习）")
    print("=" * 50)

if __name__ == '__main__':
    create_dataset_split()

