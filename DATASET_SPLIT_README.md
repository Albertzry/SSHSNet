# 数据集划分说明

## 概述

`create_dataset_split.py` 脚本用于为 SSHSNet 生成数据集划分文件。

## 数据集目录结构

要求的数据集目录结构：

```
dataset/
├── imagesTr/      # 训练集图像（.nii.gz格式）
├── labelsTr/      # 训练集标签（.nii.gz格式）
├── imagesTs/      # 测试集图像（用于半监督学习的未标注数据）
└── labelsTs/      # 测试集标签（可选，推理时使用）
```

## 生成的文件

脚本会生成两个 `.pkl` 文件：

1. **splitdataset.pkl**: 包含5折交叉验证的划分
   - 结构：列表，包含5个fold
   - 每个fold：[训练集文件列表, 验证集文件列表]
   - 用于训练和验证

2. **testdataset.pkl**: 包含未标注的测试数据文件列表
   - 结构：简单的文件列表
   - 用于半监督学习

## 使用方法

```bash
python create_dataset_split.py
```

## 输出示例

```
正在读取训练集文件...
找到 481 个训练样本
找到 121 个测试样本（用于半监督学习）

Fold 1:
  训练集: 384 个样本
  验证集: 97 个样本

...

splitdataset.pklulas 已保存
testdataset.pkl 已保存

数据集划分完成！
```

## 注意事项

1. **文件名匹配**: 训练集的图像和标签文件名必须完全一致
   - 如果某个图像文件没有对应的标签文件，会被跳过并显示警告

2. **随机种子**: 默认使用 `random_state=2021` 确保结果可复现
   - 如需修改，请编辑脚本中的 `kf = KFold(n_splits=5, shuffle=True, random_state=2021)`

3. **交叉验证**: 训练数据会自动划分为5折，每个fold包含大约80%训练数据和20%验证数据

4. **测试数据**: 测试集中的文件仅用于半监督学习，不参与交叉验证

## 与原有项目的对接

生成的文件与原项目的数据加载方式完全兼容：

- `train2d_semi_supervised.py`: 使用 `splitdataset.pkl` 和 `testdataset.pkl`
- `train2D3D_concate.py`: 使用 `splitdataset.pkl`
- `inference2d.py` 和 `inference3d.py`: 使用 `splitdataset.pkl`
- `evaluate.py`: 使用 `splitdataset.pkl`

## 数据统计

- 训练集：481 个样本（带标签）
- 测试集：121 个样本（用于半监督学习）
- 交叉验证：5 折

