#!/usr/bin/env python
"""
SSHSNet 统一执行脚本
用于从项目根目录执行各种训练、推理和评估任务

Usage:
    python run_all.py --task <task_name> [other arguments]
    
Available tasks:
    - split: 生成数据集划分文件
    - preprocess_2d_train: 预处理2D训练数据
    - preprocess_2d_test: 预处理2D测试数据
    - preprocess_3d_train: 预处理3D训练数据
    - train_2d: 训练2D网络
    - train_3d: 训练3D网络
    - infer_2d: 2D网络推理
    - eval_2d: 2D网络评估
    - infer_3d: 3D网络推理
    - eval_3d: 3D网络评估
    - predict: 测试集预测
"""

import sys
import os
import subprocess

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_command(cmd, description):
    """运行命令并打印描述"""
    print(f"\n{'='*80}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*80)
    return subprocess.run(cmd, cwd=project_root)

def preprocess_2d_train(args):
    """预处理2D训练数据"""
    cmd = [
        'python', 'data_loading/process_data.py',
        '--filepath', './dataset/imagesTr',
        '--maskpath', './dataset/labelsTr',
        '--savepath', './dataset/processdata2D',
        '--process2D', 'True',
        '--withlabel', 'True',
        '--infomation', 'info.csv'
    ]
    return run_command(cmd, "预处理2D训练数据")

def preprocess_2d_test(args):
    """预处理2D测试数据"""
    cmd = [
        'python', 'data_loading/process_data.py',
        '--filepath', './dataset/imagesTs',
        '--savepath', './dataset/processdata2D',
        '--process2D', 'True',
        '--infomation', 'unlabel_info.csv'
    ]
    return run_command(cmd, "预处理2D测试数据")

def preprocess_3d_train(args):
    """预处理3D训练数据"""
    cmd = [
        'python', 'data_loading/process_data.py',
        '--filepath', './dataset/imagesTr',
        '--maskpath', './dataset/labelsTr',
        '--savepath', './dataset/processdata3D',
        '--withlabel', 'True',
        '--infomation', 'info.csv'
    ]
    return run_command(cmd, "预处理3D训练数据")

def train_2d(args):
    """训练2D网络"""
    fold = getattr(args, 'fold', 0)
    gpuid = getattr(args, 'gpuid', '0')
    exid = getattr(args, 'exid', 'ex0')
    datapath = getattr(args, 'datapath', './dataset/processdata2D')
    train_batch_size = getattr(args, 'train_batch_size', 8)
    seed = getattr(args, 'seed', 100)
    
    cmd = [
        'python', 'training/train2d_semi_supervised.py',
        '--fold', str(fold),
        '--gpuid', gpuid,
        '--exid', exid,
        '--datapath', datapath,
        '--train_batch_size', str(train_batch_size),
        '--seed', str(seed)
    ]
    return run_command(cmd, f"训练2D网络 (fold={fold})")

def train_3d(args):
    """训练3D网络"""
    fold = getattr(args, 'fold', 0)
    gpuid = getattr(args, 'gpuid', '0,1')
    exid = getattr(args, 'exid', 'ex1')
    exid2d = getattr(args, 'exid2d', 'ex0')
    datapath = getattr(args, 'datapath', './dataset/processdata3D')
    seed = getattr(args, 'seed', 100)
    
    cmd = [
        'python', 'training/train2D3D_concate.py',
        '--fold', str(fold),
        '--gpuid', gpuid,
        '--exid', exid,
        '--exid2D', exid2d,
        '--datapath', datapath,
        '--seed', str(seed)
    ]
    return run_command(cmd, f"训练3D网络 (fold={fold})")

def infer_2d(args):
    """2D网络推理"""
    fold = getattr(args, 'fold', 0)
    gpu = getattr(args, 'gpu', '0')
    ex = getattr(args, 'ex', 'ex0')
    mainpath = getattr(args, 'mainpath', './dataset/processdata2D/')
    standerpath = getattr(args, 'standerpath', './dataset/labelsTr')
    
    cmd = [
        'python', 'inference/inference2d.py',
        '--fold', str(fold),
        '--gpu', gpu,
        '--ex', ex,
        '--mainpath', mainpath,
        '--infomation', 'info.csv',
        '--standerpath', standerpath
    ]
    return run_command(cmd, f"2D网络推理 (fold={fold})")

def eval_2d(args):
    """2D网络评估"""
    fold = getattr(args, 'fold', 0)
    exid = getattr(args, 'exid', 'ex0')
    standerpath = getattr(args, 'standerpath', './dataset/labelsTr')
    
    cmd = [
        'python', 'evaluation/evaluate.py',
        '--fold', str(fold),
        '--exid', exid,
        '--standerpath', standerpath
    ]
    return run_command(cmd, f"2D网络评估 (fold={fold})")

def infer_3d(args):
    """3D网络推理"""
    fold = getattr(args, 'fold', 0)
    gpu = getattr(args, 'gpu', '0,1')
    ex = getattr(args, 'ex', 'ex1')
    mainpath = getattr(args, 'mainpath', './dataset/processdata3D/')
    standerpath = getattr(args, 'standerpath', './dataset/labelsTr')
    
    cmd = [
        'python', 'inference/inference3d.py',
        '--fold', str(fold),
        '--gpu', gpu,
        '--ex', ex,
        '--mainpath', mainpath,
        '--infomation', 'info.csv',
        '--standerpath', standerpath
    ]
    return run_command(cmd, f"3D网络推理 (fold={fold})")

def eval_3d(args):
    """3D网络评估"""
    fold = getattr(args, 'fold', 0)
    exid = getattr(args, 'exid', 'ex1')
    standerpath = getattr(args, 'standerpath', './dataset/labelsTr')
    
    cmd = [
        'python', 'evaluation/evaluate.py',
        '--fold', str(fold),
        '--exid', exid,
        '--standerpath', standerpath
    ]
    return run_command(cmd, f"3D网络评估 (fold={fold})")

def predict(args):
    """测试集预测"""
    gpu = getattr(args, 'gpu', '0,1')
    exid2d = getattr(args, 'exid2d', 'ex0')
    exid3d = getattr(args, 'exid3d', 'ex1')
    datapath = getattr(args, 'datapath', './dataset/processdata3D_test')
    oridatapath = getattr(args, 'oridatapath', './dataset/imagesTs')
    batch_size = getattr(args, 'batch_size', 20)
    
    cmd = [
        'python', 'inference/predict_fivefold.py',
        '--gpu', gpu,
        '--exid2D', exid2d,
        '--exid3D', exid3d,
        '--datapath', datapath,
        '--oridatapath', oridatapath,
        '--batch_size', str(batch_size),
        '--infomation', 'info.csv'
    ]
    return run_command(cmd, "测试集预测")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SSHSNet 统一执行脚本')
    parser.add_argument('--task', type=str, required=True,
                       choices=['split', 'preprocess_2d_train', 'preprocess_2d_test', 'preprocess_3d_train',
                               'train_2d', 'train_3d', 'infer_2d', 'eval_2d', 'infer_3d', 'eval_3d', 'predict'],
                       help='要执行的任务')
    
    # 添加所有可能的参数
    parser.add_argument('--fold', type=int, default=0, help='fold number (0-4)')
    parser.add_argument('--gpuid', type=str, default='0', help='GPU ID for training 2D')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID for inference')
    parser.add_argument('--exid', type=str, default='ex0', help='experiment ID')
    parser.add_argument('--exid2d', type=str, default='ex0', help='2D experiment ID')
    parser.add_argument('--exid3d', type=str, default='ex1', help='3D experiment ID')
    parser.add_argument('--ex', type=str, default='ex0', help='experiment ID for inference')
    parser.add_argument('--datapath', type=str, default='./dataset/processdata2D', help='data path')
    parser.add_argument('--mainpath', type=str, default='./dataset/processdata2D/', help='main path')
    parser.add_argument('--standerpath', type=str, default='./dataset/labelsTr', help='standard path')
    parser.add_argument('--oridatapath', type=str, default='./dataset/imagesTs', help='original data path')
    parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size for prediction')
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    
    args = parser.parse_args()
    
    # 任务映射
    task_map = {
        'split': lambda args: run_command(['python', 'configs/create_dataset_split.py'], "生成数据集划分"),
        'preprocess_2d_train': preprocess_2d_train,
        'preprocess_2d_test': preprocess_2d_test,
        'preprocess_3d_train': preprocess_3d_train,
        'train_2d': train_2d,
        'train_3d': train_3d,
        'infer_2d': infer_2d,
        'eval_2d': eval_2d,
        'infer_3d': infer_3d,
        'eval_3d': eval_3d,
        'predict': predict,
    }
    
    # 执行任务
    result = task_map[args.task](args)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()

