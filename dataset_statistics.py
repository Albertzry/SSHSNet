#!/usr/bin/env python3
"""
数据集统计脚本
用于分析医学图像数据集的标注情况
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
from datetime import datetime

# 可选依赖
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed, progress bar disabled")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, visualization disabled")

class DatasetStatistics:
    def __init__(self, mr_dir, mask_dir):
        """
        初始化数据集统计器
        
        Args:
            mr_dir: MR图像文件夹路径
            mask_dir: Mask标注文件夹路径
        """
        self.mr_dir = Path(mr_dir)
        self.mask_dir = Path(mask_dir)
        self.statistics = []
        
    def find_mask_for_image(self, image_name):
        """
        为给定的图像文件查找对应的mask文件
        
        Args:
            image_name: 图像文件名，例如 'Case10.nii.gz'
            
        Returns:
            mask文件的完整路径，如果未找到则返回None
        """
        # 尝试几种可能的命名模式
        base_name = image_name.replace('Case', 'mask_case').replace('.nii.gz', '')
        possible_names = [
            f"{base_name}.nii.gz",
            f"mask_{image_name}",
            image_name.replace('Case', 'mask_Case'),
        ]
        
        for name in possible_names:
            mask_path = self.mask_dir / name
            if mask_path.exists():
                return mask_path
        return None
    
    def analyze_single_case(self, image_path, mask_path):
        """
        分析单个病例的统计信息
        
        Args:
            image_path: 图像文件路径
            mask_path: mask文件路径
            
        Returns:
            包含统计信息的字典
        """
        try:
            # 读取图像和mask
            img_nib = nib.load(str(image_path))
            img_data = img_nib.get_fdata()
            
            mask_nib = nib.load(str(mask_path))
            mask_data = mask_nib.get_fdata()
            
            # 基本信息
            case_name = image_path.name
            img_shape = img_data.shape
            mask_shape = mask_data.shape
            
            # 检查形状是否匹配
            shape_match = (img_shape == mask_shape)
            
            # 获取体素间距（分辨率）
            img_spacing = img_nib.header.get_zooms()
            mask_spacing = mask_nib.header.get_zooms()
            
            # 计算实际物理尺寸 (mm)
            physical_size = tuple(s * sp for s, sp in zip(img_shape, img_spacing))
            
            # 统计标注信息
            unique_labels = np.unique(mask_data)
            num_classes = len(unique_labels)
            
            # 计算每个类别的体素数量
            label_counts = {}
            label_percentages = {}
            total_voxels = mask_data.size
            
            for label in unique_labels:
                count = np.sum(mask_data == label)
                label_counts[int(label)] = int(count)
                label_percentages[int(label)] = (count / total_voxels) * 100
            
            # 标注体素总数（假设0是背景）
            annotated_voxels = int(np.sum(mask_data > 0))
            annotation_ratio = (annotated_voxels / total_voxels) * 100
            
            # 图像强度统计
            img_min = float(np.min(img_data))
            img_max = float(np.max(img_data))
            img_mean = float(np.mean(img_data))
            img_std = float(np.std(img_data))
            
            # 在标注区域内的图像强度统计
            if annotated_voxels > 0:
                annotated_region = img_data[mask_data > 0]
                roi_mean = float(np.mean(annotated_region))
                roi_std = float(np.std(annotated_region))
                roi_min = float(np.min(annotated_region))
                roi_max = float(np.max(annotated_region))
            else:
                roi_mean = roi_std = roi_min = roi_max = 0.0
            
            return {
                'case_name': case_name,
                'has_mask': True,
                'shape_match': shape_match,
                'img_shape_0': img_shape[0],
                'img_shape_1': img_shape[1],
                'img_shape_2': img_shape[2],
                'img_spacing_0': float(img_spacing[0]),
                'img_spacing_1': float(img_spacing[1]),
                'img_spacing_2': float(img_spacing[2]),
                'physical_size_0_mm': float(physical_size[0]),
                'physical_size_1_mm': float(physical_size[1]),
                'physical_size_2_mm': float(physical_size[2]),
                'total_voxels': total_voxels,
                'num_classes': num_classes,
                'unique_labels': str(list(unique_labels)),
                'annotated_voxels': annotated_voxels,
                'annotation_ratio_%': annotation_ratio,
                'label_counts': str(label_counts),
                'label_percentages': str(label_percentages),
                'img_min': img_min,
                'img_max': img_max,
                'img_mean': img_mean,
                'img_std': img_std,
                'roi_mean': roi_mean,
                'roi_std': roi_std,
                'roi_min': roi_min,
                'roi_max': roi_max,
            }
            
        except Exception as e:
            return {
                'case_name': image_path.name,
                'has_mask': False,
                'error': str(e)
            }
    
    def run_analysis(self):
        """
        运行完整的数据集分析
        """
        print("="*80)
        print("数据集统计分析")
        print("="*80)
        print(f"MR图像目录: {self.mr_dir}")
        print(f"Mask标注目录: {self.mask_dir}")
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # 获取所有MR图像文件
        image_files = sorted(list(self.mr_dir.glob("*.nii.gz")))
        print(f"\n找到 {len(image_files)} 个MR图像文件")
        
        # 逐个分析
        matched_cases = 0
        unmatched_cases = []
        
        # 使用tqdm或普通循环
        iterator = tqdm(image_files, desc="分析进度") if HAS_TQDM else image_files
        for img_path in iterator:
            mask_path = self.find_mask_for_image(img_path.name)
            
            if mask_path is None:
                unmatched_cases.append(img_path.name)
                continue
            
            stats = self.analyze_single_case(img_path, mask_path)
            if stats.get('has_mask', False):
                self.statistics.append(stats)
                matched_cases += 1
        
        print(f"\n成功匹配并分析: {matched_cases} 个病例")
        if unmatched_cases:
            print(f"未找到对应mask的图像: {len(unmatched_cases)} 个")
            print("未匹配的文件:", unmatched_cases[:10], "..." if len(unmatched_cases) > 10 else "")
        
        return matched_cases > 0
    
    def generate_summary(self):
        """
        生成统计摘要
        """
        if not self.statistics:
            print("没有可用的统计数据")
            return None
        
        df = pd.DataFrame(self.statistics)
        
        print("\n" + "="*80)
        print("统计摘要")
        print("="*80)
        
        print(f"\n1. 数据集基本信息:")
        print(f"   - 总病例数: {len(df)}")
        print(f"   - 形状匹配的病例: {df['shape_match'].sum()} ({df['shape_match'].sum()/len(df)*100:.1f}%)")
        
        print(f"\n2. 图像尺寸统计:")
        print(f"   - Shape 维度0: {df['img_shape_0'].min():.0f} ~ {df['img_shape_0'].max():.0f} (平均: {df['img_shape_0'].mean():.1f})")
        print(f"   - Shape 维度1: {df['img_shape_1'].min():.0f} ~ {df['img_shape_1'].max():.0f} (平均: {df['img_shape_1'].mean():.1f})")
        print(f"   - Shape 维度2: {df['img_shape_2'].min():.0f} ~ {df['img_shape_2'].max():.0f} (平均: {df['img_shape_2'].mean():.1f})")
        
        print(f"\n3. 体素间距统计 (mm):")
        print(f"   - Spacing 维度0: {df['img_spacing_0'].min():.3f} ~ {df['img_spacing_0'].max():.3f} (平均: {df['img_spacing_0'].mean():.3f})")
        print(f"   - Spacing 维度1: {df['img_spacing_1'].min():.3f} ~ {df['img_spacing_1'].max():.3f} (平均: {df['img_spacing_1'].mean():.3f})")
        print(f"   - Spacing 维度2: {df['img_spacing_2'].min():.3f} ~ {df['img_spacing_2'].max():.3f} (平均: {df['img_spacing_2'].mean():.3f})")
        
        print(f"\n4. 物理尺寸统计 (mm):")
        print(f"   - 物理尺寸 维度0: {df['physical_size_0_mm'].min():.1f} ~ {df['physical_size_0_mm'].max():.1f} (平均: {df['physical_size_0_mm'].mean():.1f})")
        print(f"   - 物理尺寸 维度1: {df['physical_size_1_mm'].min():.1f} ~ {df['physical_size_1_mm'].max():.1f} (平均: {df['physical_size_1_mm'].mean():.1f})")
        print(f"   - 物理尺寸 维度2: {df['physical_size_2_mm'].min():.1f} ~ {df['physical_size_2_mm'].max():.1f} (平均: {df['physical_size_2_mm'].mean():.1f})")
        
        print(f"\n5. 标注统计:")
        print(f"   - 类别数量: {df['num_classes'].min():.0f} ~ {df['num_classes'].max():.0f} (平均: {df['num_classes'].mean():.1f})")
        print(f"   - 标注体素数: {df['annotated_voxels'].min():.0f} ~ {df['annotated_voxels'].max():.0f} (平均: {df['annotated_voxels'].mean():.0f})")
        print(f"   - 标注占比%: {df['annotation_ratio_%'].min():.2f}% ~ {df['annotation_ratio_%'].max():.2f}% (平均: {df['annotation_ratio_%'].mean():.2f}%)")
        
        print(f"\n6. 图像强度统计:")
        print(f"   - 最小值: {df['img_min'].min():.1f} ~ {df['img_min'].max():.1f} (平均: {df['img_min'].mean():.1f})")
        print(f"   - 最大值: {df['img_max'].min():.1f} ~ {df['img_max'].max():.1f} (平均: {df['img_max'].mean():.1f})")
        print(f"   - 平均值: {df['img_mean'].min():.1f} ~ {df['img_mean'].max():.1f} (平均: {df['img_mean'].mean():.1f})")
        print(f"   - 标准差: {df['img_std'].min():.1f} ~ {df['img_std'].max():.1f} (平均: {df['img_std'].mean():.1f})")
        
        print(f"\n7. 标注区域(ROI)强度统计:")
        print(f"   - ROI平均值: {df['roi_mean'].min():.1f} ~ {df['roi_mean'].max():.1f} (平均: {df['roi_mean'].mean():.1f})")
        print(f"   - ROI标准差: {df['roi_std'].min():.1f} ~ {df['roi_std'].max():.1f} (平均: {df['roi_std'].mean():.1f})")
        
        # 分析所有病例的标签分布
        all_labels = set()
        for labels_str in df['unique_labels']:
            labels = eval(labels_str)
            all_labels.update(labels)
        
        print(f"\n8. 数据集中所有出现的标签: {sorted(all_labels)}")
        
        return df
    
    def save_results(self, output_dir="./dataset_analysis"):
        """
        保存分析结果
        
        Args:
            output_dir: 输出目录
        """
        if not self.statistics:
            print("没有可用的统计数据")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.statistics)
        
        # 保存详细的CSV文件
        csv_path = output_path / "dataset_statistics_detailed.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n详细统计数据已保存到: {csv_path}")
        
        # 保存摘要统计
        summary_stats = df.describe()
        summary_path = output_path / "dataset_statistics_summary.csv"
        summary_stats.to_csv(summary_path)
        print(f"摘要统计已保存到: {summary_path}")
        
        # 生成可视化图表
        self.create_visualizations(df, output_path)
        
        # 生成Markdown报告
        self.generate_markdown_report(df, output_path)
    
    def generate_markdown_report(self, df, output_path):
        """
        生成详细的Markdown格式统计报告
        
        Args:
            df: 统计数据DataFrame
            output_path: 输出路径
        """
        print("\n生成Markdown报告...")
        
        md_path = output_path / "dataset_statistics_report.md"
        
        with open(md_path, 'w', encoding='utf-8') as f:
            # 标题和基本信息
            f.write("# 数据集统计报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**MR图像目录**: `{self.mr_dir}`\n\n")
            f.write(f"**Mask标注目录**: `{self.mask_dir}`\n\n")
            f.write("---\n\n")
            
            # 1. 数据集概览
            f.write("## 1. 数据集概览\n\n")
            f.write(f"- **总病例数**: {len(df)}\n")
            f.write(f"- **形状匹配的病例**: {df['shape_match'].sum()} ({df['shape_match'].sum()/len(df)*100:.1f}%)\n")
            f.write(f"- **数据完整性**: {'✓ 所有病例都有对应的mask标注' if df['has_mask'].all() else '✗ 部分病例缺少mask标注'}\n\n")
            
            # 2. 图像尺寸统计
            f.write("## 2. 图像尺寸统计\n\n")
            f.write("### 2.1 体素尺寸分布\n\n")
            f.write("| 维度 | 最小值 | 最大值 | 平均值 | 标准差 | 中位数 |\n")
            f.write("|------|--------|--------|--------|--------|--------|\n")
            f.write(f"| 维度0 (Z轴) | {df['img_shape_0'].min():.0f} | {df['img_shape_0'].max():.0f} | {df['img_shape_0'].mean():.1f} | {df['img_shape_0'].std():.1f} | {df['img_shape_0'].median():.1f} |\n")
            f.write(f"| 维度1 (Y轴) | {df['img_shape_1'].min():.0f} | {df['img_shape_1'].max():.0f} | {df['img_shape_1'].mean():.1f} | {df['img_shape_1'].std():.1f} | {df['img_shape_1'].median():.1f} |\n")
            f.write(f"| 维度2 (X轴) | {df['img_shape_2'].min():.0f} | {df['img_shape_2'].max():.0f} | {df['img_shape_2'].mean():.1f} | {df['img_shape_2'].std():.1f} | {df['img_shape_2'].median():.1f} |\n\n")
            
            # 2.2 体素间距统计
            f.write("### 2.2 体素间距 (分辨率) 统计 (mm)\n\n")
            f.write("| 维度 | 最小值 | 最大值 | 平均值 | 标准差 | 中位数 |\n")
            f.write("|------|--------|--------|--------|--------|--------|\n")
            f.write(f"| 维度0 | {df['img_spacing_0'].min():.4f} | {df['img_spacing_0'].max():.4f} | {df['img_spacing_0'].mean():.4f} | {df['img_spacing_0'].std():.4f} | {df['img_spacing_0'].median():.4f} |\n")
            f.write(f"| 维度1 | {df['img_spacing_1'].min():.4f} | {df['img_spacing_1'].max():.4f} | {df['img_spacing_1'].mean():.4f} | {df['img_spacing_1'].std():.4f} | {df['img_spacing_1'].median():.4f} |\n")
            f.write(f"| 维度2 | {df['img_spacing_2'].min():.4f} | {df['img_spacing_2'].max():.4f} | {df['img_spacing_2'].mean():.4f} | {df['img_spacing_2'].std():.4f} | {df['img_spacing_2'].median():.4f} |\n\n")
            
            # 2.3 物理尺寸统计
            f.write("### 2.3 物理尺寸统计 (mm)\n\n")
            f.write("| 维度 | 最小值 | 最大值 | 平均值 | 标准差 | 中位数 |\n")
            f.write("|------|--------|--------|--------|--------|--------|\n")
            f.write(f"| 维度0 | {df['physical_size_0_mm'].min():.1f} | {df['physical_size_0_mm'].max():.1f} | {df['physical_size_0_mm'].mean():.1f} | {df['physical_size_0_mm'].std():.1f} | {df['physical_size_0_mm'].median():.1f} |\n")
            f.write(f"| 维度1 | {df['physical_size_1_mm'].min():.1f} | {df['physical_size_1_mm'].max():.1f} | {df['physical_size_1_mm'].mean():.1f} | {df['physical_size_1_mm'].std():.1f} | {df['physical_size_1_mm'].median():.1f} |\n")
            f.write(f"| 维度2 | {df['physical_size_2_mm'].min():.1f} | {df['physical_size_2_mm'].max():.1f} | {df['physical_size_2_mm'].mean():.1f} | {df['physical_size_2_mm'].std():.1f} | {df['physical_size_2_mm'].median():.1f} |\n\n")
            
            # 3. 标注统计
            f.write("## 3. 标注统计\n\n")
            f.write("### 3.1 标注基本信息\n\n")
            f.write("| 统计项 | 最小值 | 最大值 | 平均值 | 标准差 | 中位数 |\n")
            f.write("|--------|--------|--------|--------|--------|--------|\n")
            f.write(f"| 类别数量 | {df['num_classes'].min():.0f} | {df['num_classes'].max():.0f} | {df['num_classes'].mean():.1f} | {df['num_classes'].std():.1f} | {df['num_classes'].median():.1f} |\n")
            f.write(f"| 标注体素数 | {df['annotated_voxels'].min():,.0f} | {df['annotated_voxels'].max():,.0f} | {df['annotated_voxels'].mean():,.0f} | {df['annotated_voxels'].std():,.0f} | {df['annotated_voxels'].median():,.0f} |\n")
            f.write(f"| 标注占比 (%) | {df['annotation_ratio_%'].min():.2f} | {df['annotation_ratio_%'].max():.2f} | {df['annotation_ratio_%'].mean():.2f} | {df['annotation_ratio_%'].std():.2f} | {df['annotation_ratio_%'].median():.2f} |\n")
            f.write(f"| 总体素数 | {df['total_voxels'].min():,.0f} | {df['total_voxels'].max():,.0f} | {df['total_voxels'].mean():,.0f} | {df['total_voxels'].std():,.0f} | {df['total_voxels'].median():,.0f} |\n\n")
            
            # 3.2 标签分布统计
            f.write("### 3.2 标签分布统计\n\n")
            all_labels = set()
            for labels_str in df['unique_labels']:
                labels = eval(labels_str)
                all_labels.update(labels)
            
            f.write(f"**数据集中出现的所有标签**: {sorted(all_labels)}\n\n")
            f.write(f"**标签总数**: {len(all_labels)} 个 (包括背景标签0)\n\n")
            f.write(f"**实际标注类别数**: {len(all_labels) - 1} 个 (不包括背景)\n\n")
            
            # 统计每个标签在所有病例中的出现频率
            label_frequency = {}
            label_total_counts = {}
            label_total_percentages = []
            
            for label in all_labels:
                label_frequency[label] = 0
                label_total_counts[label] = 0
            
            for idx, row in df.iterrows():
                labels = eval(row['unique_labels'])
                label_counts = eval(row['label_counts'])
                label_percentages = eval(row['label_percentages'])
                
                for label in labels:
                    label_frequency[label] += 1
                    if label in label_counts:
                        label_total_counts[label] += label_counts[label]
                        label_total_percentages.append((label, label_percentages[label]))
            
            f.write("#### 3.2.1 标签出现频率\n\n")
            f.write("| 标签 | 出现次数 | 出现频率 (%) | 在所有病例中的平均占比 (%) |\n")
            f.write("|------|----------|--------------|----------------------------|\n")
            
            for label in sorted(all_labels):
                freq = label_frequency[label]
                freq_pct = (freq / len(df)) * 100
                # 计算该标签在所有出现该标签的病例中的平均占比
                label_pcts = [pct for lbl, pct in label_total_percentages if lbl == label]
                avg_pct = np.mean(label_pcts) if label_pcts else 0.0
                f.write(f"| {label:.0f} | {freq} | {freq_pct:.1f} | {avg_pct:.2f} |\n")
            
            f.write("\n")
            
            # 3.3 标注分布详细分析
            f.write("### 3.3 标注分布详细分析\n\n")
            
            # 找出标注最多的病例
            max_annotated_idx = df['annotated_voxels'].idxmax()
            min_annotated_idx = df['annotated_voxels'].idxmin()
            max_ratio_idx = df['annotation_ratio_%'].idxmax()
            min_ratio_idx = df['annotation_ratio_%'].idxmin()
            
            f.write(f"- **标注体素数最多的病例**: {df.loc[max_annotated_idx, 'case_name']} ({df.loc[max_annotated_idx, 'annotated_voxels']:,} 体素, {df.loc[max_annotated_idx, 'annotation_ratio_%']:.2f}%)\n")
            f.write(f"- **标注体素数最少的病例**: {df.loc[min_annotated_idx, 'case_name']} ({df.loc[min_annotated_idx, 'annotated_voxels']:,} 体素, {df.loc[min_annotated_idx, 'annotation_ratio_%']:.2f}%)\n")
            f.write(f"- **标注占比最高的病例**: {df.loc[max_ratio_idx, 'case_name']} ({df.loc[max_ratio_idx, 'annotation_ratio_%']:.2f}%, {df.loc[max_ratio_idx, 'annotated_voxels']:,} 体素)\n")
            f.write(f"- **标注占比最低的病例**: {df.loc[min_ratio_idx, 'case_name']} ({df.loc[min_ratio_idx, 'annotation_ratio_%']:.2f}%, {df.loc[min_ratio_idx, 'annotated_voxels']:,} 体素)\n\n")
            
            # 4. 图像强度统计
            f.write("## 4. 图像强度统计\n\n")
            f.write("### 4.1 全图像强度统计\n\n")
            f.write("| 统计项 | 最小值 | 最大值 | 平均值 | 标准差 | 中位数 |\n")
            f.write("|--------|--------|--------|--------|--------|--------|\n")
            f.write(f"| 最小值 | {df['img_min'].min():.1f} | {df['img_min'].max():.1f} | {df['img_min'].mean():.1f} | {df['img_min'].std():.1f} | {df['img_min'].median():.1f} |\n")
            f.write(f"| 最大值 | {df['img_max'].min():.1f} | {df['img_max'].max():.1f} | {df['img_max'].mean():.1f} | {df['img_max'].std():.1f} | {df['img_max'].median():.1f} |\n")
            f.write(f"| 平均值 | {df['img_mean'].min():.1f} | {df['img_mean'].max():.1f} | {df['img_mean'].mean():.1f} | {df['img_mean'].std():.1f} | {df['img_mean'].median():.1f} |\n")
            f.write(f"| 标准差 | {df['img_std'].min():.1f} | {df['img_std'].max():.1f} | {df['img_std'].mean():.1f} | {df['img_std'].std():.1f} | {df['img_std'].median():.1f} |\n\n")
            
            f.write("### 4.2 标注区域 (ROI) 强度统计\n\n")
            f.write("| 统计项 | 最小值 | 最大值 | 平均值 | 标准差 | 中位数 |\n")
            f.write("|--------|--------|--------|--------|--------|--------|\n")
            f.write(f"| ROI平均值 | {df['roi_mean'].min():.1f} | {df['roi_mean'].max():.1f} | {df['roi_mean'].mean():.1f} | {df['roi_mean'].std():.1f} | {df['roi_mean'].median():.1f} |\n")
            f.write(f"| ROI标准差 | {df['roi_std'].min():.1f} | {df['roi_std'].max():.1f} | {df['roi_std'].mean():.1f} | {df['roi_std'].std():.1f} | {df['roi_std'].median():.1f} |\n")
            f.write(f"| ROI最小值 | {df['roi_min'].min():.1f} | {df['roi_min'].max():.1f} | {df['roi_min'].mean():.1f} | {df['roi_min'].std():.1f} | {df['roi_min'].median():.1f} |\n")
            f.write(f"| ROI最大值 | {df['roi_max'].min():.1f} | {df['roi_max'].max():.1f} | {df['roi_max'].mean():.1f} | {df['roi_max'].std():.1f} | {df['roi_max'].median():.1f} |\n\n")
            
            # 5. 数据质量分析
            f.write("## 5. 数据质量分析\n\n")
            
            # 检查形状一致性
            shape_consistency = {}
            for col in ['img_shape_0', 'img_shape_1', 'img_shape_2']:
                unique_shapes = df[col].unique()
                shape_consistency[col] = len(unique_shapes)
            
            f.write("### 5.1 尺寸一致性\n\n")
            f.write(f"- **维度0唯一尺寸数**: {shape_consistency['img_shape_0']} 种\n")
            f.write(f"- **维度1唯一尺寸数**: {shape_consistency['img_shape_1']} 种\n")
            f.write(f"- **维度2唯一尺寸数**: {shape_consistency['img_shape_2']} 种\n\n")
            
            if shape_consistency['img_shape_0'] == 1 and shape_consistency['img_shape_1'] == 1:
                f.write("✓ 所有病例在维度0和维度1上尺寸一致\n\n")
            else:
                f.write("⚠ 病例在维度0和维度1上存在尺寸不一致的情况\n\n")
            
            # 检查间距一致性
            spacing_consistency = {}
            for col in ['img_spacing_0', 'img_spacing_1', 'img_spacing_2']:
                unique_spacings = df[col].nunique()
                spacing_consistency[col] = unique_spacings
            
            f.write("### 5.2 体素间距一致性\n\n")
            f.write(f"- **维度0唯一间距数**: {spacing_consistency['img_spacing_0']} 种\n")
            f.write(f"- **维度1唯一间距数**: {spacing_consistency['img_spacing_1']} 种\n")
            f.write(f"- **维度2唯一间距数**: {spacing_consistency['img_spacing_2']} 种\n\n")
            
            # 6. 可视化图表
            f.write("## 6. 可视化图表\n\n")
            f.write("以下图表展示了数据集的统计分布:\n\n")
            f.write("1. **数据集概览图** (`dataset_statistics_overview.png`)\n")
            f.write("   - 图像尺寸分布\n")
            f.write("   - 类别数量分布\n")
            f.write("   - 标注占比分布\n")
            f.write("   - 标注体素数与占比关系\n\n")
            f.write("2. **体素间距分布图** (`voxel_spacing_distribution.png`)\n")
            f.write("   - 三个维度的体素间距分布\n\n")
            f.write("3. **强度分布图** (`intensity_distribution.png`)\n")
            f.write("   - 全图像强度统计\n")
            f.write("   - ROI区域强度统计\n\n")
            
            # 7. 数据文件
            f.write("## 7. 生成的数据文件\n\n")
            f.write("| 文件名 | 描述 |\n")
            f.write("|--------|------|\n")
            f.write("| `dataset_statistics_detailed.csv` | 每个病例的详细统计数据 |\n")
            f.write("| `dataset_statistics_summary.csv` | 统计摘要（均值、标准差等） |\n")
            f.write("| `dataset_statistics_report.md` | 本报告文件 |\n")
            f.write("| `dataset_statistics_overview.png` | 数据集概览可视化图表 |\n")
            f.write("| `voxel_spacing_distribution.png` | 体素间距分布图 |\n")
            f.write("| `intensity_distribution.png` | 强度分布图 |\n\n")
            
            # 8. 总结
            f.write("## 8. 总结\n\n")
            f.write(f"本数据集包含 **{len(df)}** 个病例，所有病例都有对应的mask标注。\n\n")
            
            # 获取最常见的尺寸
            mode_0 = df['img_shape_0'].mode()
            mode_1 = df['img_shape_1'].mode()
            mode_2 = df['img_shape_2'].mode()
            shape_0_str = f"{mode_0.iloc[0]:.0f}" if len(mode_0) > 0 else f"{df['img_shape_0'].median():.0f}"
            shape_1_str = f"{mode_1.iloc[0]:.0f}" if len(mode_1) > 0 else f"{df['img_shape_1'].median():.0f}"
            shape_2_str = f"{mode_2.iloc[0]:.0f}" if len(mode_2) > 0 else f"{df['img_shape_2'].median():.0f}"
            
            f.write(f"- 图像尺寸主要集中在 **{shape_0_str} × {shape_1_str} × {shape_2_str}** 体素\n")
            f.write(f"- 平均每个病例包含 **{df['num_classes'].mean():.1f}** 个不同的标注类别\n")
            f.write(f"- 平均标注占比为 **{df['annotation_ratio_%'].mean():.2f}%**\n")
            f.write(f"- 数据集中共出现 **{len(all_labels)}** 个不同的标签值（0-{max(all_labels):.0f}）\n\n")
            
            f.write("---\n\n")
            f.write(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"Markdown报告已保存到: {md_path}")
    
    def create_visualizations(self, df, output_path):
        """
        创建可视化图表
        
        Args:
            df: 统计数据DataFrame
            output_path: 输出路径
        """
        if not HAS_MATPLOTLIB:
            print("\n跳过可视化：matplotlib未安装")
            return
            
        print("\n生成可视化图表...")
        
        # 设置样式
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 图像尺寸分布
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Dataset Statistics Overview', fontsize=16, fontweight='bold')
        
        # Shape分布
        axes[0, 0].hist(df['img_shape_0'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Image Shape - Dimension 0')
        axes[0, 0].set_xlabel('Voxels')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(df['img_shape_1'], bins=20, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Image Shape - Dimension 1')
        axes[0, 1].set_xlabel('Voxels')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[0, 2].hist(df['img_shape_2'], bins=20, color='lightcoral', edgecolor='black')
        axes[0, 2].set_title('Image Shape - Dimension 2')
        axes[0, 2].set_xlabel('Voxels')
        axes[0, 2].set_ylabel('Frequency')
        
        # 标注统计
        axes[1, 0].hist(df['num_classes'], bins=range(int(df['num_classes'].min()), int(df['num_classes'].max())+2), 
                       color='orange', edgecolor='black')
        axes[1, 0].set_title('Number of Classes per Case')
        axes[1, 0].set_xlabel('Number of Classes')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(df['annotation_ratio_%'], bins=30, color='purple', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Annotation Ratio Distribution')
        axes[1, 1].set_xlabel('Annotation Ratio (%)')
        axes[1, 1].set_ylabel('Frequency')
        
        axes[1, 2].scatter(df['annotated_voxels'], df['annotation_ratio_%'], alpha=0.5, color='teal')
        axes[1, 2].set_title('Annotated Voxels vs Ratio')
        axes[1, 2].set_xlabel('Annotated Voxels')
        axes[1, 2].set_ylabel('Annotation Ratio (%)')
        
        plt.tight_layout()
        viz_path = output_path / "dataset_statistics_overview.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"可视化图表已保存到: {viz_path}")
        
        # 2. 体素间距分布
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Voxel Spacing Distribution (mm)', fontsize=14, fontweight='bold')
        
        axes[0].hist(df['img_spacing_0'], bins=20, color='skyblue', edgecolor='black')
        axes[0].set_title('Spacing - Dimension 0')
        axes[0].set_xlabel('Spacing (mm)')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(df['img_spacing_1'], bins=20, color='lightgreen', edgecolor='black')
        axes[1].set_title('Spacing - Dimension 1')
        axes[1].set_xlabel('Spacing (mm)')
        axes[1].set_ylabel('Frequency')
        
        axes[2].hist(df['img_spacing_2'], bins=20, color='lightcoral', edgecolor='black')
        axes[2].set_title('Spacing - Dimension 2')
        axes[2].set_xlabel('Spacing (mm)')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        spacing_path = output_path / "voxel_spacing_distribution.png"
        plt.savefig(spacing_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"体素间距分布图已保存到: {spacing_path}")
        
        # 3. 图像强度分布
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Image Intensity Statistics', fontsize=14, fontweight='bold')
        
        axes[0, 0].hist(df['img_mean'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Image Mean Intensity')
        axes[0, 0].set_xlabel('Mean Intensity')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(df['img_std'], bins=30, color='indianred', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Image Std Intensity')
        axes[0, 1].set_xlabel('Std Intensity')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].hist(df['roi_mean'], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('ROI Mean Intensity')
        axes[1, 0].set_xlabel('Mean Intensity')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(df['roi_std'], bins=30, color='darkorange', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('ROI Std Intensity')
        axes[1, 1].set_xlabel('Std Intensity')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        intensity_path = output_path / "intensity_distribution.png"
        plt.savefig(intensity_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"强度分布图已保存到: {intensity_path}")


def main():
    """
    主函数
    """
    # 设置数据路径
    mr_dir = "/root/SSHSNet/dataset/MR"
    mask_dir = "/root/SSHSNet/dataset/Mask"
    output_dir = "/root/SSHSNet/dataset_analysis"
    
    # 创建统计器
    stats = DatasetStatistics(mr_dir, mask_dir)
    
    # 运行分析
    success = stats.run_analysis()
    
    if success:
        # 生成摘要
        df = stats.generate_summary()
        
        # 保存结果
        stats.save_results(output_dir)
        
        print("\n" + "="*80)
        print("分析完成！")
        print(f"所有结果已保存到: {output_dir}")
        print("="*80)
    else:
        print("\n分析失败：没有找到有效的数据")


if __name__ == "__main__":
    main()

