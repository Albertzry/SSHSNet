# SSHSNet 快速开始指南

本指南提供使用统一执行脚本 `run_all.py` 的快速入门方法。

## 基本用法

```bash
python run_all.py --task <task_name> [其他参数]
```

## 可用任务

### 1. 生成数据集划分
```bash
python run_all.py --task split
```

### 2. 预处理数据

#### 预处理2D训练数据
```bash
python run_all.py --task preprocess_2d_train
```

#### 预处理2D测试数据
```bash
python run_all.py --task preprocess_2d_test
```

#### 预处理3D训练数据
```bash
python run_all.py --task preprocess_3d_train
```

### 3. 训练网络

#### 训练2D网络（fold 0）
```bash
python run_all.py --task train_2d --fold 0 --gpuid 1 --exid ex0
```

#### 训练3D网络（fold 0）
```bash
python run_all.py --task train_3d --fold 0 --gpuid 1 --exid ex1 --exid2d ex0
```

### 4. 推理和评估

#### 2D网络推理
```bash
python run_all.py --task infer_2d --fold 0 --gpu 0 --ex ex0
```

#### 2D网络评估
```bash
python run_all.py --task eval_2d --fold 0 --exid ex0
```

#### 3D网络推理
```bash
python run_all.py --task infer_3d --fold 0 --gpu 0,1 --ex ex1
```

#### 3D网络评估
```bash
python run_all.py --task eval_3d --fold 0 --exid ex1
```

### 5. 测试集预测
```bash
python run_all.py --task predict --gpu 0,1 --exid2d ex0 --exid3d ex1
```

## 参数说明

### 常用参数

- `--fold`: fold编号 (0-4)，默认0
- `--gpuid`: 2D训练使用的GPU ID，默认'0'
- `--gpu`: 推理使用的GPU ID，默认'0'
- `--exid`: 实验ID，默认'ex0'
- `--exid2d`: 2D实验ID，默认'ex0'
- `--exid3d`: 3D实验ID，默认'ex1'
- `--ex`: 推理实验ID，默认'ex0'
- `--datapath`: 数据路径
- `--standerpath`: 标准标签路径
- `--train_batch_size`: 训练batch大小，默认8
- `--batch_size`: 预测batch大小，默认20
- `--seed`: 随机种子，默认100

## 完整工作流示例

### 步骤1: 数据准备
```bash
# 生成数据集划分
python run_all.py --task split

# 预处理2D数据
python run_all.py --task preprocess_2d_train
python run_all.py --task preprocess_2d_test

# 预处理3D数据
python run_all.py --task preprocess_3d_train
```

### 步骤2: 训练（5折交叉验证）
```bash
# 训练2D网络（所有fold）
for fold in 0 1 2 3 4; do
    python run_all.py --task train_2d --fold ${fold} --gpuid 0 --exid ex0
done

# 训练3D网络（所有fold）
for fold in 0 1 2 3 4; do
    python run_all.py --task train_3d --fold ${fold} --gpuid 0,1 --exid ex1 --exid2d ex0
done
```

### 步骤3: 评估
```bash
# 评估2D网络（所有fold）
for fold in 0 1 2 3 4; do
    python run_all.py --task infer_2d --fold ${fold} --gpu 0 --ex ex0
    python run_all.py --task eval_2d --fold ${fold} --exid ex0
done

# 评估3D网络（所有fold）
for fold in 0 1 2 3 4; do
    python run_all.py --task infer_3d --fold ${fold} --gpu 0,1 --ex ex1
    python run_all.py --task eval_3d --fold ${fold} --exid ex1
done
```

### 步骤4: 测试集预测
```bash
python run_all.py --task predict --gpu 0,1 --exid2d ex0 --exid3d ex1
```

## 注意事项

1. **首次使用**: 确保已完成环境配置（见README.md）
2. **GPU设置**: 根据实际GPU配置调整 `--gpuid` 和 `--gpu` 参数
3. **内存管理**: 如遇内存不足，调整 `--train_batch_size` 参数
4. **路径**: 脚本会自动从项目根目录执行，无需cd到子目录

## 查看更多

- 完整命令列表和详细参数说明: 见 `docs/EXECUTION_COMMANDS.md`
- 项目结构说明: 见 `docs/REFACTORING_SUMMARY.md`

