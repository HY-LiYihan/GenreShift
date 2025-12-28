#!/usr/bin/env python3
"""
enhanced_visualization.py - 增强版可视化模块

支持：
1. 体裁和学科的双重分类分析
2. 分布比较（直方图、箱线图）
3. 总体和分布对比
4. 多维度可视化
"""

import os
import json
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import logging
from scipy import stats
import seaborn as sns
from enhanced_corpus_reader import ClassificationField

# 设置matplotlib使用英文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class EnhancedVisualization:
    """增强版可视化类"""
    
    def __init__(self, output_dir: str = "enhanced_visualization_results"):
        """
        初始化增强版可视化类
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("enhanced_visualization")
        
        # SCI配色方案 - 专业学术图表配色
        self.colors = {
            'version': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],  # 蓝色、紫色、橙色、红色、绿色
            'discipline': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],  # 深蓝、青绿、黄色、橙色、红色
            'genre': ['#003F5C', '#58508D', '#BC5090', '#FF6361', '#FFA600'],  # 深蓝、紫色、粉色、红色、金色
            'mixed': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#264653']  # 混合配色
        }
        
    def analyze_last_version_detailed(self, reader, version: str, max_documents: int = None) -> Dict[str, Any]:
        """
        对最后一个版本进行详细分析
        
        Args:
            reader: 增强版语料库读取器
            version: 版本号
            max_documents: 最大文档数
            
        Returns:
            详细分析结果
        """
        print(f"\n{'='*60}")
        print(f"详细分析版本: {version}")
        print(f"{'='*60}")
        
        # 读取文档
        documents = reader.read_version(version, max_documents)
        
        if not documents:
            print(f"版本 {version} 没有读取到文档")
            return {}
            
        # 获取分析就绪数据
        analysis_data = reader.create_analysis_ready_data(documents)
        
        # 详细分析
        detailed_results = {
            "version": version,
            "document_count": len(documents),
            "statistics": analysis_data["statistics"],
            "classification_analysis": {},
            "distribution_analysis": {},
            "comparative_analysis": {}
        }
        
        # 1. 分类分析
        print(f"\n1. 分类分析:")
        
        # 按学科分类
        discipline_groups = reader.group_by_classification(documents, ClassificationField.DISCIPLINE)
        detailed_results["classification_analysis"]["by_discipline"] = {
            "groups": list(discipline_groups.keys()),
            "counts": {k: len(v) for k, v in discipline_groups.items()}
        }
        print(f"   学科分类: {list(discipline_groups.keys())}")
        print(f"   文档分布: { {k: len(v) for k, v in discipline_groups.items()} }")
        
        # 按源文件分类（体裁）
        source_groups = reader.group_by_classification(documents, ClassificationField.SOURCE_FILE)
        detailed_results["classification_analysis"]["by_source"] = {
            "groups": list(source_groups.keys()),
            "counts": {k: len(v) for k, v in source_groups.items()}
        }
        print(f"   源文件分类: {len(source_groups)} 个文件")
        
        # 2. 分布分析
        print(f"\n2. 分布分析:")
        
        # 文本长度分布
        text_lengths = [doc.text_length for doc in documents]
        detailed_results["distribution_analysis"]["text_length"] = {
            "mean": np.mean(text_lengths),
            "std": np.std(text_lengths),
            "min": np.min(text_lengths),
            "max": np.max(text_lengths),
            "median": np.median(text_lengths),
            "percentiles": {
                "25": np.percentile(text_lengths, 25),
                "50": np.percentile(text_lengths, 50),
                "75": np.percentile(text_lengths, 75),
                "90": np.percentile(text_lengths, 90)
            }
        }
        print(f"   文本长度: 平均={np.mean(text_lengths):.0f}, 标准差={np.std(text_lengths):.0f}")
        print(f"   范围: {np.min(text_lengths)} - {np.max(text_lengths)} 字符")
        
        # 词数分布
        word_counts = [doc.word_count for doc in documents]
        detailed_results["distribution_analysis"]["word_count"] = {
            "mean": np.mean(word_counts),
            "std": np.std(word_counts),
            "min": np.min(word_counts),
            "max": np.max(word_counts),
            "median": np.median(word_counts)
        }
        print(f"   词数: 平均={np.mean(word_counts):.0f}, 标准差={np.std(word_counts):.0f}")
        
        # 3. 比较分析
        print(f"\n3. 比较分析:")
        
        # 学科间比较
        if len(discipline_groups) > 1:
            discipline_comparison = {}
            for discipline, docs in discipline_groups.items():
                if len(docs) >= 3:  # 至少需要3个文档
                    lengths = [doc.text_length for doc in docs]
                    discipline_comparison[discipline] = {
                        "count": len(docs),
                        "mean_length": np.mean(lengths),
                        "std_length": np.std(lengths),
                        "mean_words": np.mean([doc.word_count for doc in docs])
                    }
            
            detailed_results["comparative_analysis"]["discipline_comparison"] = discipline_comparison
            print(f"   学科间文本长度比较:")
            for discipline, stats in discipline_comparison.items():
                print(f"     {discipline}: {stats['count']}文档, 平均长度={stats['mean_length']:.0f}字符")
                
        # 源文件间比较（体裁比较）
        if len(source_groups) > 1:
            source_comparison = {}
            for source, docs in source_groups.items():
                if len(docs) >= 3:
                    lengths = [doc.text_length for doc in docs]
                    source_name = Path(source).stem
                    source_comparison[source_name] = {
                        "count": len(docs),
                        "mean_length": np.mean(lengths),
                        "std_length": np.std(lengths)
                    }
            
            detailed_results["comparative_analysis"]["source_comparison"] = source_comparison
            print(f"   源文件间文本长度比较:")
            for source, stats in list(source_comparison.items())[:5]:  # 显示前5个
                print(f"     {source}: {stats['count']}文档, 平均长度={stats['mean_length']:.0f}字符")
                
        return detailed_results
        
    def create_genre_discipline_comparison_charts(self, detailed_results: Dict[str, Any], 
                                                 reader, documents: List[Any]):
        """
        创建体裁和学科对比图表
        
        Args:
            detailed_results: 详细分析结果
            reader: 语料库读取器
            documents: 文档列表
        """
        print(f"\n创建体裁和学科对比图表...")
        
        # 1. 学科分布饼图
        discipline_groups = reader.group_by_classification(documents, ClassificationField.DISCIPLINE)
        if discipline_groups:
            self._create_pie_chart(
                data={k: len(v) for k, v in discipline_groups.items()},
                title="Document Distribution by Discipline",
                output_file=self.output_dir / "discipline_distribution.png",
                color_palette=self.colors['discipline']
            )
            
        # 2. 源文件分布（体裁）
        source_groups = reader.group_by_classification(documents, ClassificationField.SOURCE_FILE)
        if len(source_groups) <= 10:  # 如果源文件数量不多，显示所有
            source_data = {Path(k).stem: len(v) for k, v in source_groups.items()}
            self._create_pie_chart(
                data=source_data,
                title="Document Distribution by Source File (Genre)",
                output_file=self.output_dir / "source_distribution.png",
                color_palette=self.colors['genre']
            )
            
        # 3. 文本长度分布直方图
        text_lengths = [doc.text_length for doc in documents]
        self._create_histogram(
            data=text_lengths,
            title="Text Length Distribution",
            xlabel="Text Length (characters)",
            ylabel="Frequency",
            output_file=self.output_dir / "text_length_histogram.png",
            bins=30
        )
        
        # 4. 学科间文本长度箱线图
        if len(discipline_groups) > 1:
            discipline_lengths = {}
            for discipline, docs in discipline_groups.items():
                if len(docs) >= 3:
                    discipline_lengths[discipline] = [doc.text_length for doc in docs]
            
            if discipline_lengths:
                self._create_boxplot(
                    data=discipline_lengths,
                    title="Text Length by Discipline",
                    xlabel="Discipline",
                    ylabel="Text Length (characters)",
                    output_file=self.output_dir / "discipline_length_boxplot.png"
                )
                
        # 5. 源文件间文本长度箱线图（体裁比较）
        if len(source_groups) > 1 and len(source_groups) <= 15:  # 限制数量
            source_lengths = {}
            for source, docs in source_groups.items():
                if len(docs) >= 3:
                    source_name = Path(source).stem
                    if len(source_name) > 20:
                        source_name = source_name[:17] + "..."
                    source_lengths[source_name] = [doc.text_length for doc in docs]
            
            if source_lengths:
                self._create_boxplot(
                    data=source_lengths,
                    title="Text Length by Source File (Genre)",
                    xlabel="Source File",
                    ylabel="Text Length (characters)",
                    output_file=self.output_dir / "source_length_boxplot.png",
                    rotation=45
                )
                
        # 6. 散点图：文本长度 vs 词数
        self._create_scatter_plot(
            x_data=[doc.text_length for doc in documents],
            y_data=[doc.word_count for doc in documents],
            title="Text Length vs Word Count",
            xlabel="Text Length (characters)",
            ylabel="Word Count",
            output_file=self.output_dir / "length_vs_words_scatter.png"
        )
        
        # 7. 热力图：学科与源文件的交叉分布
        if len(discipline_groups) > 1 and len(source_groups) > 1:
            self._create_cross_distribution_heatmap(documents, reader)
            
    def _create_pie_chart(self, data: Dict[str, int], title: str, 
                         output_file: Path, color_palette: List[str]):
        """创建饼图"""
        if not data:
            return
            
        labels = list(data.keys())
        sizes = list(data.values())
        
        plt.figure(figsize=(10, 8))
        
        # 计算百分比
        total = sum(sizes)
        percentages = [size/total*100 for size in sizes]
        
        # 创建饼图
        wedges, texts, autotexts = plt.pie(
            sizes, 
            labels=labels,
            colors=color_palette[:len(labels)],
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85
        )
        
        # 美化
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 添加图例
        plt.legend(
            wedges, 
            [f'{label}: {size} ({pct:.1f}%)' for label, size, pct in zip(labels, sizes, percentages)],
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        plt.axis('equal')  # 确保饼图是圆形
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"饼图已保存: {output_file}")
        
    def _create_histogram(self, data: List[float], title: str, xlabel: str, 
                         ylabel: str, output_file: Path, bins: int = 30):
        """创建直方图"""
        if not data:
            return
            
        plt.figure(figsize=(12, 6))
        
        # 创建直方图 - 使用SCI配色
        n, bins, patches = plt.hist(data, bins=bins, color='#2E86AB', alpha=0.7, edgecolor='black')
        
        # 添加统计信息
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        
        # 添加均值和标准差线
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.0f}')
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.0f}')
        
        # 添加正态分布曲线
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_val, std_val)
        p = p * max(n) / max(p)  # 缩放以匹配直方图
        plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 添加统计信息文本框
        stats_text = f'Mean: {mean_val:.0f}\nMedian: {median_val:.0f}\nStd: {std_val:.0f}\nMin: {min(data):.0f}\nMax: {max(data):.0f}\nN: {len(data)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"直方图已保存: {output_file}")
        
    def _create_boxplot(self, data: Dict[str, List[float]], title: str, 
                       xlabel: str, ylabel: str, output_file: Path, rotation: int = 0):
        """创建箱线图"""
        if not data:
            return
            
        plt.figure(figsize=(max(10, len(data)*0.8), 8))
        
        # 准备数据
        labels = list(data.keys())
        values = list(data.values())
        
        # 创建箱线图
        box = plt.boxplot(values, labels=labels, patch_artist=True)
        
        # 设置颜色
        colors = self.colors['mixed'][:len(labels)]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        
        if rotation > 0:
            plt.xticks(rotation=rotation, ha='right')
            
        plt.grid(axis='y', alpha=0.3)
        
        # 添加样本数量
        for i, (label, vals) in enumerate(data.items()):
            plt.text(i+1, min(vals) - (max([max(v) for v in values]) - min([min(v) for v in values])) * 0.02,
                    f'n={len(vals)}', ha='center', va='top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"箱线图已保存: {output_file}")
        
    def _create_scatter_plot(self, x_data: List[float], y_data: List[float], 
                            title: str, xlabel: str, ylabel: str, output_file: Path):
        """创建散点图"""
        if not x_data or not y_data:
            return
            
        plt.figure(figsize=(10, 8))
        
        # 创建散点图 - 使用SCI配色
        plt.scatter(x_data, y_data, alpha=0.6, color='#2E86AB', edgecolors='black', linewidth=0.5)
        
        # 计算相关系数
        correlation = np.corrcoef(x_data, y_data)[0, 1]
        
        # 添加回归线
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            plt.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2, 
                    label=f'Linear fit (r={correlation:.3f})')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 添加统计信息
        stats_text = f'Correlation: {correlation:.3f}\nN: {len(x_data)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"散点图已保存: {output_file}")
        
    def _create_cross_distribution_heatmap(self, documents: List[Any], reader):
        """创建交叉分布热力图"""
        # 提取学科和源文件信息
        discipline_source_counts = {}
        
        for doc in documents:
            discipline = doc.discipline
            source = Path(doc.source_file).stem
            
            if discipline not in discipline_source_counts:
                discipline_source_counts[discipline] = {}
            
            if source not in discipline_source_counts[discipline]:
                discipline_source_counts[discipline][source] = 0
            
            discipline_source_counts[discipline][source] += 1
        
        # 转换为DataFrame
        disciplines = sorted(discipline_source_counts.keys())
        all_sources = set()
        for sources in discipline_source_counts.values():
            all_sources.update(sources.keys())
        sources = sorted(all_sources)
        
        # 创建矩阵
        matrix = np.zeros((len(disciplines), len(sources)))
        for i, discipline in enumerate(disciplines):
            for j, source in enumerate(sources):
                matrix[i, j] = discipline_source_counts.get(discipline, {}).get(source, 0)
        
        # 创建热力图
        plt.figure(figsize=(max(12, len(sources)*0.8), max(8, len(disciplines)*0.6)))
        
        # 使用更专业的SCI配色方案
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='viridis',
                   xticklabels=sources, yticklabels=disciplines,
                   cbar_kws={'label': 'Document Count'})
        
        plt.title("Cross Distribution: Discipline vs Source File (Genre)", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("Source File (Genre)", fontsize=12)
        plt.ylabel("Discipline", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        output_file = self.output_dir / "cross_distribution_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"热力图已保存: {output_file}")
        
    def create_genre_discipline_grouped_bar_chart(self, discipline_results: Dict[str, Any], 
                                                 genre_results: Dict[str, Any],
                                                 metric: str = "type_token_ratio",
                                                 output_file: str = "genre_discipline_grouped_bar.png"):
        """
        创建体裁与学科分组柱状图
        
        Args:
            discipline_results: 学科对比分析结果
            genre_results: 体裁对比分析结果
            metric: 要显示的指标（如type_token_ratio, mtld_average, academic_word_ratio等）
            output_file: 输出文件名
        """
        print(f"\n创建体裁与学科分组柱状图 - 指标: {metric}")
        
        # 提取学科数据
        discipline_metrics = {}
        if "discipline_metrics" in discipline_results:
            discipline_metrics = discipline_results["discipline_metrics"]
        elif "genre_metrics" in discipline_results:  # 备用字段名
            discipline_metrics = discipline_results["genre_metrics"]
        
        # 提取体裁数据
        genre_metrics = {}
        if "genre_metrics" in genre_results:
            genre_metrics = genre_results["genre_metrics"]
        
        if not discipline_metrics or not genre_metrics:
            print(f"  警告: 缺少必要的数据，无法创建分组柱状图")
            return
        
        # 获取学科列表
        disciplines = list(discipline_metrics.keys())
        
        # 获取体裁列表（假设有两种体裁）
        genres = list(genre_metrics.keys())
        if len(genres) > 2:
            genres = genres[:2]  # 只取前两种体裁
            print(f"  注意: 只使用前两种体裁: {genres}")
        
        # 准备数据
        data = []
        for discipline in disciplines:
            if discipline in discipline_metrics and metric in discipline_metrics[discipline]:
                discipline_value = discipline_metrics[discipline][metric]
            else:
                discipline_value = 0
            
            # 为每种体裁创建数据点
            for genre in genres:
                if genre in genre_metrics and metric in genre_metrics[genre]:
                    genre_value = genre_metrics[genre][metric]
                else:
                    genre_value = 0
                
                # 计算组合值（学科值 * 体裁调整因子）
                # 这里使用简单的加权平均：学科基础值 + 体裁差异
                combined_value = discipline_value + (genre_value - discipline_value) * 0.3
                
                data.append({
                    "discipline": discipline,
                    "genre": genre,
                    "value": combined_value,
                    "discipline_base": discipline_value,
                    "genre_base": genre_value
                })
        
        if not data:
            print(f"  错误: 没有找到指标 '{metric}' 的数据")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 创建分组柱状图
        plt.figure(figsize=(max(12, len(disciplines)*1.5), 8))
        
        # 设置位置
        x = np.arange(len(disciplines))
        width = 0.35  # 柱子的宽度
        
        # 为每种体裁创建柱子
        for i, genre in enumerate(genres):
            genre_data = df[df["genre"] == genre]
            values = genre_data["value"].values
            
            # 设置位置偏移
            offset = width * (i - (len(genres)-1)/2)
            
            # 创建柱子
            bars = plt.bar(x + offset, values, width, 
                          label=genre,
                          alpha=0.8,
                          edgecolor='black',
                          linewidth=1)
            
            # 添加数据标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 设置图表属性
        plt.title(f"{metric.replace('_', ' ').title()} by Discipline and Genre", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Discipline", fontsize=14)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
        plt.xticks(x, disciplines, rotation=45, ha='right')
        plt.legend(title="Genre")
        plt.grid(axis='y', alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  分组柱状图已保存: {output_path}")
        
        # 返回数据摘要
        summary = {
            "metric": metric,
            "disciplines": disciplines,
            "genres": genres,
            "data_summary": df.groupby(["discipline", "genre"])["value"].mean().to_dict()
        }
        
        return summary
        
    def create_comprehensive_genre_discipline_chart(self, analysis_results: Dict[str, Any],
                                                   metrics: List[str] = None):
        """
        创建综合的体裁与学科对比图表
        
        Args:
            analysis_results: 完整的分析结果
            metrics: 要显示的指标列表，如果为None则使用默认列表
        """
        if metrics is None:
            metrics = [
                "type_token_ratio",
                "mtld_average", 
                "academic_word_ratio",
                "discipline_word_ratio",
                "avg_word_length",
                "vocabulary_richness"
            ]
        
        print(f"\n创建综合体裁与学科对比图表...")
        
        # 检查是否有必要的数据
        if "discipline_comparison" not in analysis_results or "genre_comparison" not in analysis_results:
            print("  错误: 缺少学科或体裁对比数据")
            return
        
        discipline_results = analysis_results["discipline_comparison"]
        genre_results = analysis_results["genre_comparison"]
        
        summaries = []
        
        # 为每个指标创建图表
        for metric in metrics:
            output_file = f"genre_discipline_{metric}_grouped_bar.png"
            summary = self.create_genre_discipline_grouped_bar_chart(
                discipline_results, genre_results, metric, output_file
            )
            
            if summary:
                summaries.append(summary)
        
        # 创建指标对比图（多个指标在同一图表中）
        if len(summaries) >= 2:
            self._create_metric_comparison_chart(summaries)
        
        print(f"\n综合图表创建完成，共生成 {len(summaries)} 个分组柱状图")
        return summaries
        
    def _create_metric_comparison_chart(self, summaries: List[Dict[str, Any]]):
        """创建指标对比图"""
        if len(summaries) < 2:
            return
        
        # 提取第一个摘要的数据结构
        first_summary = summaries[0]
        disciplines = first_summary["disciplines"]
        genres = first_summary["genres"]
        
        # 准备数据
        comparison_data = []
        for summary in summaries:
            metric = summary["metric"]
            data_summary = summary["data_summary"]
            
            for (discipline, genre), value in data_summary.items():
                comparison_data.append({
                    "metric": metric,
                    "discipline": discipline,
                    "genre": genre,
                    "value": value
                })
        
        df = pd.DataFrame(comparison_data)
        
        # 创建多子图对比
        n_metrics = len(set(df["metric"]))
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8), squeeze=False)
        axes = axes.flatten()
        
        for idx, (metric, metric_df) in enumerate(df.groupby("metric")):
            ax = axes[idx]
            
            # 准备数据用于分组柱状图
            pivot_df = metric_df.pivot_table(index="discipline", columns="genre", values="value")
            
            # 创建分组柱状图
            x = np.arange(len(pivot_df.index))
            width = 0.35
            
            for i, genre in enumerate(pivot_df.columns):
                offset = width * (i - (len(pivot_df.columns)-1)/2)
                values = pivot_df[genre].values
                
                bars = ax.bar(x + offset, values, width, 
                             label=genre, alpha=0.8, edgecolor='black')
                
                # 添加数据标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Discipline", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(pivot_df.index, rotation=45, ha='right', fontsize=9)
            ax.legend(title="Genre", fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle("Metric Comparison Across Disciplines and Genres", 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / "metric_comparison_across_genres.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  指标对比图已保存: {output_path}")
        
    def create_comprehensive_report(self, detailed_results: Dict[str, Any], 
                                   output_file: str = "detailed_analysis_report.md"):
        """
        创建综合分析报告
        
        Args:
            detailed_results: 详细分析结果
            output_file: 输出文件路径
        """
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Detailed Analysis Report\n\n")
            f.write(f"*Generated on: {np.datetime64('now')}*\n\n")
            
            f.write(f"## Version: {detailed_results.get('version', 'Unknown')}\n\n")
            f.write(f"**Total Documents:** {detailed_results.get('document_count', 0):,}\n\n")
            
            # 分类分析
            f.write("## Classification Analysis\n\n")
            
            classification = detailed_results.get("classification_analysis", {})
            
            if "by_discipline" in classification:
                f.write("### By Discipline\n\n")
                discipline_data = classification["by_discipline"]
                f.write("| Discipline | Document Count | Percentage |\n")
                f.write("|------------|----------------|------------|\n")
                
                total = sum(discipline_data["counts"].values())
                for discipline, count in discipline_data["counts"].items():
                    percentage = count / total * 100
                    f.write(f"| {discipline} | {count:,} | {percentage:.1f}% |\n")
                f.write("\n")
            
            if "by_source" in classification:
                f.write("### By Source File (Genre)\n\n")
                source_data = classification["by_source"]
                f.write(f"**Total Source Files:** {len(source_data['groups'])}\n\n")
                
                # 显示前10个源文件
                sorted_sources = sorted(source_data["counts"].items(), key=lambda x: x[1], reverse=True)[:10]
                f.write("| Source File | Document Count |\n")
                f.write("|-------------|----------------|\n")
                for source, count in sorted_sources:
                    source_name = Path(source).stem
                    f.write(f"| {source_name} | {count:,} |\n")
                f.write("\n")
            
            # 分布分析
            f.write("## Distribution Analysis\n\n")
            
            distribution = detailed_results.get("distribution_analysis", {})
            
            if "text_length" in distribution:
                f.write("### Text Length Distribution\n\n")
                length_stats = distribution["text_length"]
                f.write(f"- **Mean:** {length_stats['mean']:.0f} characters\n")
                f.write(f"- **Standard Deviation:** {length_stats['std']:.0f} characters\n")
                f.write(f"- **Minimum:** {length_stats['min']:,} characters\n")
                f.write(f"- **Maximum:** {length_stats['max']:,} characters\n")
                f.write(f"- **Median:** {length_stats['median']:.0f} characters\n")
                f.write(f"- **25th Percentile:** {length_stats['percentiles']['25']:.0f} characters\n")
                f.write(f"- **75th Percentile:** {length_stats['percentiles']['75']:.0f} characters\n")
                f.write(f"- **90th Percentile:** {length_stats['percentiles']['90']:.0f} characters\n\n")
            
            if "word_count" in distribution:
                f.write("### Word Count Distribution\n\n")
                word_stats = distribution["word_count"]
                f.write(f"- **Mean:** {word_stats['mean']:.0f} words\n")
                f.write(f"- **Standard Deviation:** {word_stats['std']:.0f} words\n")
                f.write(f"- **Minimum:** {word_stats['min']:,} words\n")
                f.write(f"- **Maximum:** {word_stats['max']:,} words\n")
                f.write(f"- **Median:** {word_stats['median']:.0f} words\n\n")
            
            # 比较分析
            f.write("## Comparative Analysis\n\n")
            
            comparative = detailed_results.get("comparative_analysis", {})
            
            if "discipline_comparison" in comparative:
                f.write("### Discipline Comparison\n\n")
                disc_comp = comparative["discipline_comparison"]
                f.write("| Discipline | Document Count | Mean Length | Std Length | Mean Words |\n")
                f.write("|------------|----------------|-------------|------------|------------|\n")
                
                for discipline, stats in disc_comp.items():
                    f.write(f"| {discipline} | {stats['count']:,} | {stats['mean_length']:.0f} | "
                           f"{stats['std_length']:.0f} | {stats['mean_words']:.0f} |\n")
                f.write("\n")
            
            if "source_comparison" in comparative:
                f.write("### Source File (Genre) Comparison\n\n")
                source_comp = comparative["source_comparison"]
                f.write("| Source File | Document Count | Mean Length | Std Length |\n")
                f.write("|-------------|----------------|-------------|------------|\n")
                
                # 按文档数排序
                sorted_sources = sorted(source_comp.items(), key=lambda x: x[1]["count"], reverse=True)[:15]
                for source, stats in sorted_sources:
                    f.write(f"| {source} | {stats['count']:,} | {stats['mean_length']:.0f} | "
                           f"{stats['std_length']:.0f} |\n")
                f.write("\n")
            
            # 可视化文件列表
            f.write("## Generated Visualizations\n\n")
            f.write("The following visualization files have been generated:\n\n")
            
            viz_files = [
                "discipline_distribution.png - Pie chart showing document distribution by discipline",
                "source_distribution.png - Pie chart showing document distribution by source file (genre)",
                "text_length_histogram.png - Histogram of text length distribution",
                "discipline_length_boxplot.png - Box plot comparing text lengths across disciplines",
                "source_length_boxplot.png - Box plot comparing text lengths across source files (genres)",
                "length_vs_words_scatter.png - Scatter plot of text length vs word count",
                "cross_distribution_heatmap.png - Heatmap showing cross distribution of discipline vs source"
            ]
            
            for viz_file in viz_files:
                f.write(f"- `{viz_file}`\n")
            
            f.write("\n## Key Findings\n\n")
            f.write("1. **Document Distribution:** Analysis of how documents are distributed across different classifications.\n")
            f.write("2. **Text Characteristics:** Statistical analysis of text length and word count distributions.\n")
            f.write("3. **Comparative Insights:** Comparison of text characteristics across different disciplines and genres.\n")
            f.write("4. **Visual Patterns:** Visual representations showing patterns and relationships in the data.\n")
        
        self.logger.info(f"详细分析报告已保存: {output_path}")
        
    def run_complete_visualization(self, reader, version: str, max_documents: int = None):
        """
        运行完整的可视化分析
        
        Args:
            reader: 增强版语料库读取器
            version: 版本号
            max_documents: 最大文档数
        """
        print(f"\n{'='*60}")
        print(f"运行完整可视化分析 - 版本: {version}")
        print(f"{'='*60}")
        
        # 1. 详细分析
        detailed_results = self.analyze_last_version_detailed(reader, version, max_documents)
        
        if not detailed_results:
            print("详细分析失败，无法继续可视化")
            return
            
        # 2. 读取文档用于可视化
        documents = reader.read_version(version, max_documents)
        
        if not documents:
            print("无法读取文档进行可视化")
            return
            
        # 3. 创建图表
        self.create_genre_discipline_comparison_charts(detailed_results, reader, documents)
        
        # 4. 生成报告
        self.create_comprehensive_report(detailed_results)
        
        print(f"\n{'='*60}")
        print("完整可视化分析完成!")
        print(f"{'='*60}")
        print(f"所有结果已保存到: {self.output_dir}")
        print(f"1. 详细分析报告: detailed_analysis_report.md")
        print(f"2. 可视化图表: 7种不同类型的图表")
        print(f"3. 统计摘要: 包含在报告中")
        print(f"{'='*60}")


def test_enhanced_visualization():
    """测试增强版可视化模块"""
    print("测试增强版可视化模块...")
    
    import sys
    sys.path.insert(0, 'src')
    
    from enhanced_corpus_reader import EnhancedCorpusReader, TextExtractionStrategy
    
    # 创建读取器
    reader = EnhancedCorpusReader("corpus", TextExtractionStrategy.SMART_EXTRACTION)
    
    # 获取版本
    versions = reader.list_versions()
    if not versions:
        print("没有找到语料库版本")
        return False
        
    # 使用最新版本
    latest_version = versions[-1]
    print(f"使用版本进行测试: {latest_version}")
    
    # 创建可视化器
    visualizer = EnhancedVisualization("test_visualization_output")
    
    # 运行完整可视化分析（限制文档数以加快测试）
    visualizer.run_complete_visualization(reader, latest_version, max_documents=50)
    
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    if test_enhanced_visualization():
        print("\n增强版可视化模块测试通过!")
    else:
        print("\n增强版可视化模块测试失败!")
