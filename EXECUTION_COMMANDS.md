# SSHSNet 执行指令指南

基于您的数据集路径 `/root/SSHSNet/dataset` 和原始README.md流程，以下是完整的执行指令。

## 环境要求

- Linux系统
- CUDA 10.1
- Python 3.6
- GeForce RTX 2080 (或兼容GPU)

### 依赖包版本
```
torch 1.7.1
scikit-image 0.17.2
scikit-learn 0.24.0
SimpleITK 2.0.2
nibabel 3.2.1
nnunet 1.6.6
numpy 1.19.4
pandas 1.1.5
argparse 1.4.0
albumentations 0.5.2
segmentation-models-pytorch 0.1.3
tensorboard 2.4.1
MedPy 0.4.0
matplotlib 3.3.2
```

## 数据集准备

### 1. 生成数据集划分文件
```bash
# 生成 splitdataset.pkl 和 testdataset.pkl
python create_dataset_split.py
```

### 2. 数据预处理

#### 处理2D训练数据（带标签）
```bash
python process_data.py \
    --filepath './dataset/imagesTr' \
    --maskpath "./dataset/labelsTr" \
    --savepath "./dataset/processdata2D" \
    --process2D True \
    --withlabel True \
    --infomation 'info.csv'
```

#### 处理2D未标注数据（用于半监督学习）
```bash
python process_data.py \
    --filepath './dataset/imagesTs' \
    --savepath "./dataset/processdata2D" \
    --process2D True \
    --infomation 'unlabel_info.csv'
```

#### 处理3D训练数据（带标签）
```bash
python process_data.py \
    --filepath './dataset/imagesTr' \
    --maskpath "./dataset/labelsTr" \
    --savepath "./dataset/processdata3D" \
    --withlabel True \
    --infomation 'info.csv'
```

## 训练阶段

### 1. 训练2D网络（5折交叉验证）
```bash
# 训练所有fold
for fold in 0 1 2 3 4; do
    python train2d_semi_supervised.py \
        --fold ${fold} \
        --gpuid '0' \
        --exid 'ex0' \
        --datapath "./dataset/processdata2D" \
        --train_batch_size 8 \
        --seed 2021
done
```

### 2. 训练3D网络（5折交叉验证）
```bash
# 训练所有fold
for fold in 0 1 2 3 4; do
    python train2D3D_concate.py \
        --fold ${fold} \
        --gpuid '0,1' \
        --exid 'ex1' \
        --exid2D 'ex0' \
        --datapath "./dataset/processdata3D" \
        --seed 2021
done
```

**注意**: 权重文件将保存在 `weight/ex#/sub#` 目录中。

## 推理和评估阶段

### 1. 评估2D网络（5折交叉验证）

#### 2D网络推理
```bash
for fold in 0 1 2 3 4; do
    python inference2d.py \
        --fold ${fold} \
        --gpu '0' \
        --ex 'ex0' \
        --mainpath './dataset/processdata2D/' \
        --infomation 'info.csv' \
        --standerpath './dataset/labelsTr'
done
```

#### 2D网络评估
```bash
for fold in 0 1 2 3 4; do
    python evaluate.py \
        --fold ${fold} \
        --exid 'ex0' \
        --standerpath './dataset/labelsTr'
done
```

### 2. 评估SSHSNet（5折交叉验证）

#### SSHSNet推理
```bash
for fold in 0 1 2 3 4; do
    python inference3d.py \
        --fold ${fold} \
        --gpu '0,1' \
        --ex 'ex1' \
        --mainpath './dataset/processdata3D/' \
        --infomation 'info.csv' \
        --standerpath './dataset/labelsTr'
done
```

#### SSHSNet评估
```bash
for fold in 0 1 2 3 4; do
    python evaluate.py \
        --fold ${fold} \
        --exid 'ex1' \
        --standerpath './dataset/labelsTr'
done
```

## 测试集预测

### 1. 处理测试集
```bash
python process_data.py \
    --filepath './dataset/imagesTs' \
    --savepath "./dataset/processdata3D_test" \
    --infomation 'info.csv'
```

### 2. 预测测试集
```bash
python predict_fivefold.py \
    --gpu '0,1' \
    --exid2D 'ex0' \
    --exid3D 'ex1' \
    --datapath "./dataset/processdata3D_test" \
    --oridatapath './dataset/imagesTs' \
    --batch_size 20 \
    --infomation 'info.csv'
```

**注意**: 测试集预测结果将保存在 `./dataset/processdata3D_test/ex1/predict` 路径中。

## 完整执行流程

### 步骤1: 环境准备
```bash
# 确保所有依赖包已安装
pip install torch==1.7.1 scikit-image==0.17.2 scikit-learn==0.24.0 SimpleITK==2.0.2 nibabel==3.2.1 nnunet==1.6.6 numpy==1.19.4 pandas==1.1.5 albumentations==0.5.2 segmentation-models-pytorch==0.1.3 tensorboard==2.4.1 MedPy==0.4.0 matplotlib==3.3.2
```

### 步骤2: 数据集准备
```bash
# 生成数据划分
python create_dataset_split.py

# 预处理数据
python process_data.py --filepath './dataset/imagesTr' --maskpath "./dataset/labelsTr" --savepath "./dataset/processdata2D" --process2D True --withlabel True --infomation 'info.csv'
python process_data.py --filepath './dataset/imagesTs' --savepath "./dataset/processdata2D" --process2D True --infomation 'unlabel_info.csv'
python process_data.py --filepath './dataset/imagesTr' --maskpath "./dataset/labelsTr" --savepath "./dataset/processdata3D" --withlabel True --infomation 'info.csv'
```

### 步骤3: 训练
```bash
# 训练2D网络
for fold in 0 1 2 3 4; do
    python train2d_semi_supervised.py --fold ${fold} --gpuid '0' --exid 'ex0' --datapath "./dataset/processdata2D" --train_batch_size 8 --seed 2021
done

# 训练3D网络
for fold in 0 1 2 3 4; do
    python train2D3D_concate.py --fold ${fold} --gpuid '0,1' --exid 'ex1' --exid2D 'ex0' --datapath "./dataset/processdata3D" --seed 2021
done
```

### 步骤4: 评估
```bash
# 评估2D网络
for fold in 0 1 2 3 4; do
    python inference2d.py --fold ${fold} --gpu '0' --ex 'ex0' --mainpath './dataset/processdata2D/' --infomation 'info.csv' --standerpath './dataset/labelsTr'
    python evaluate.py --fold ${fold} --exid 'ex0' --standerpath './dataset/labelsTr'
done

# 评估SSHSNet
for fold in 0 1 2 3 4; do
    python inference3d.py --fold ${fold} --gpu '0,1' --ex 'ex1' --mainpath './dataset/processdata3D/' --infomation 'info.csv' --standerpath './dataset/labelsTr'
    python evaluate.py --fold ${fold} --exid 'ex1' --standerpath './dataset/labelsTr'
done
```

### 步骤5: 测试集预测
```bash
# 处理测试集
python process_data.py --filepath './dataset/imagesTs' --savepath "./dataset/processdata3D_test" --infomation 'info.csv'

# 预测测试集
python predict_fivefold.py --gpu '0,1' --exid2D 'ex0' --exid3D 'ex1' --datapath "./dataset/processdata3D_test" --oridatapath './dataset/imagesTs' --batch_size 20 --infomation 'info.csv'
```

## 重要说明

1. **GPU设置**: 根据您的GPU配置调整 `--gpuid` 和 `--gpu` 参数
2. **内存管理**: 如果遇到内存不足，可以减小 `--train_batch_size` 和 `--batch_size`
3. **路径检查**: 确保所有路径正确，特别是数据集路径
4. **训练时间**: 完整训练可能需要数小时到数天，取决于数据量和硬件配置
5. **结果保存**: 训练权重保存在 `weight/` 目录，推理结果保存在 `log/` 目录

## 故障排除

- 如果遇到CUDA内存不足，尝试减小batch size
- 如果文件路径错误，检查数据集目录结构
- 如果依赖包版本冲突，建议使用conda环境管理
