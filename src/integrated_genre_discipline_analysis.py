#!/usr/bin/env python3
"""
integrated_genre_discipline_analysis.py - 整合版体裁与学科分析系统

整合所有功能，提供完整的体裁与学科分析，所有输出保存到以时间命名的文件夹中。
"""

import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats

# 设置matplotlib使用英文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

sys.path.insert(0, 'src')

from enhanced_corpus_reader import EnhancedCorpusReader, TextExtractionStrategy, ClassificationField
from analyzers.vocabulary_analyzer import VocabularyAnalyzer
from analyzers.enhanced_academic_analyzer import EnhancedAcademicAnalyzer
from analyzers.discipline_comparator import DisciplineComparator
from analyzers.genre_comparator import GenreComparator


class IntegratedGenreDisciplineAnalysis:
    """整合版体裁与学科分析系统"""
    
    def __init__(self, corpus_root: str = "corpus"):
        """
        初始化整合版分析系统
        
        Args:
            corpus_root: 语料库根目录路径
        """
        self.corpus_root = corpus_root
        self.reader = EnhancedCorpusReader(corpus_root, TextExtractionStrategy.SMART_EXTRACTION)
        
        # 创建输出目录（以时间命名）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("output") / f"analysis_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 分析器配置
        self.analyzer_configs = {
            "vocabulary": {
                "calculate_ttr": True,
                "calculate_mtld": True,
                "calculate_word_freq": True,
                "top_n_words": 20
            },
            "enhanced_academic": {
                "use_downloaded_wordlists": False,
                "calculate_academic_ratio": True,
                "calculate_discipline_ratios": True,
                "calculate_cross_discipline": True
            },
            "discipline_comparator": {
                "calculate_differences": True,
                "compare_pairs": True,
                "compare_to_overall": True
            },
            "genre_comparator": {
                "calculate_differences": True,
                "compare_pairs": True,
                "compare_to_overall": True,
                "calculate_statistical_significance": False,
                "genre_field": "genre"
            }
        }
        
        # SCI配色方案 - 专业学术图表配色
        self.colors = {
            'discipline': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],  # 深蓝、青绿、黄色、橙色、红色
            'genre': ['#003F5C', '#58508D', '#BC5090', '#FF6361', '#FFA600'],  # 深蓝、紫色、粉色、红色、金色
            'mixed': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#264653']  # 混合配色
        }
        
    def analyze_last_version(self, max_documents: int = None) -> Tuple[Dict[str, Any], List[Any]]:
        """
        分析最后一个版本
        
        Args:
            max_documents: 最大文档数
            
        Returns:
            (分析结果, 文档列表)
        """
        print(f"\n{'='*80}")
        print("体裁与学科分析系统")
        print(f"{'='*80}")
        
        # 获取版本
        versions = self.reader.list_versions()
        if not versions:
            print("没有找到语料库版本")
            return {}, []
            
        latest_version = versions[-1]
        print(f"分析版本: {latest_version}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*80}")
        
        # 读取文档
        print(f"\n1. 读取文档...")
        documents = self.reader.read_version(latest_version, max_documents)
        
        if not documents:
            print(f"版本 {latest_version} 没有读取到文档")
            return {}, []
            
        print(f"   成功读取 {len(documents)} 个文档")
        
        # 获取文本
        texts = self.reader.get_texts(documents)
        
        # 执行分析
        results = {
            "version": latest_version,
            "document_count": len(documents),
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(self.output_dir)
        }
        
        # 2. 分类分析
        print(f"\n2. 分类分析...")
        classification_results = self._analyze_classifications(documents)
        results.update(classification_results)
        
        # 3. 词汇分析
        print(f"\n3. 词汇分析...")
        vocab_results = self._analyze_vocabulary(texts, latest_version)
        results["vocabulary_analysis"] = vocab_results
        
        # 4. 学术词汇分析
        print(f"\n4. 学术词汇分析...")
        academic_results = self._analyze_academic_vocabulary(texts, latest_version)
        results["academic_analysis"] = academic_results
        
        # 5. 比较分析
        print(f"\n5. 比较分析...")
        comparison_results = self._analyze_comparisons(texts, documents, latest_version)
        results.update(comparison_results)
        
        return results, documents
        
    def _analyze_classifications(self, documents: List[Any]) -> Dict[str, Any]:
        """分析分类"""
        results = {}
        
        # 按学科分类
        discipline_groups = self.reader.group_by_classification(documents, ClassificationField.DISCIPLINE)
        discipline_counts = {k: len(v) for k, v in discipline_groups.items()}
        
        print(f"   学科分类 ({len(discipline_groups)} 个学科):")
        for discipline, count in discipline_counts.items():
            print(f"     {discipline}: {count} 个文档")
            
        # 检查并打印Unknown语料的详细信息
        if "Unknown" in discipline_groups:
            unknown_docs = discipline_groups["Unknown"]
            print(f"\n   发现 {len(unknown_docs)} 个Unknown语料，详细信息:")
            for i, doc in enumerate(unknown_docs[:10]):  # 最多显示10个
                print(f"     {i+1}. 文档ID: {doc.id}")
                print(f"        源文件: {doc.source_file}")
                print(f"        文件名: {doc.file_name}")
                print(f"        文本长度: {doc.text_length} 字符")
                print(f"        词数: {doc.word_count}")
                if hasattr(doc, 'title') and doc.title:
                    print(f"        标题: {doc.title[:50]}...")
            if len(unknown_docs) > 10:
                print(f"     ... 还有 {len(unknown_docs) - 10} 个Unknown语料未显示")
                
        # 按源文件分类（体裁）
        source_groups = self.reader.group_by_classification(documents, ClassificationField.SOURCE_FILE)
        source_counts = {Path(k).stem: len(v) for k, v in source_groups.items()}
        
        print(f"\n   源文件分类 ({len(source_groups)} 个文件):")
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for source, count in sorted_sources:
            print(f"     {source}: {count} 个文档")
            
        results["classification"] = {
            "disciplines": discipline_counts,
            "sources": source_counts,
            "total_disciplines": len(discipline_groups),
            "total_sources": len(source_groups)
        }
        
        # 保存Unknown语料的详细信息
        if "Unknown" in discipline_groups:
            unknown_details = []
            for doc in discipline_groups["Unknown"]:
                unknown_details.append({
                    "id": doc.id,
                    "source_file": doc.source_file,
                    "file_name": doc.file_name,
                    "text_length": doc.text_length,
                    "word_count": doc.word_count,
                    "title": doc.title if hasattr(doc, 'title') else "",
                    "version": doc.version if hasattr(doc, 'version') else ""
                })
            results["unknown_documents_details"] = unknown_details
            
        return results
        
    def _analyze_vocabulary(self, texts: List[str], version: str) -> Dict[str, Any]:
        """分析词汇"""
        vocab_analyzer = VocabularyAnalyzer(f"{version}_vocab", self.analyzer_configs["vocabulary"])
        vocab_results = vocab_analyzer.analyze(texts)
        
        if vocab_results and "error" not in vocab_results:
            stats = vocab_results["basic_statistics"]
            print(f"   总词数: {stats.get('total_tokens', 0):,}")
            print(f"   总词型: {stats.get('total_types', 0):,}")
            print(f"   TTR (词汇丰富度): {stats.get('type_token_ratio', 0):.4f}")
            
            if "mtld_analysis" in vocab_results:
                mtld = vocab_results["mtld_analysis"].get("mtld_average", 0)
                print(f"   MTLD (文本词汇多样性): {mtld:.2f}")
                
        return vocab_results if vocab_results and "error" not in vocab_results else {}
        
    def _analyze_academic_vocabulary(self, texts: List[str], version: str) -> Dict[str, Any]:
        """分析学术词汇"""
        academic_analyzer = EnhancedAcademicAnalyzer(f"{version}_academic", 
                                                    self.analyzer_configs["enhanced_academic"])
        academic_results = academic_analyzer.analyze(texts)
        
        if academic_results and "error" not in academic_results:
            if "academic_analysis" in academic_results:
                academic = academic_results["academic_analysis"]
                print(f"   总体学术词汇比例: {academic.get('overall_ratio', 0)*100:.2f}%")
                print(f"   学科词汇比例: {academic.get('discipline_ratio', 0)*100:.2f}%")
                print(f"   交叉学科词汇比例: {academic.get('cross_discipline_ratio', 0)*100:.2f}%")
                
        return academic_results if academic_results and "error" not in academic_results else {}
        
    def _analyze_comparisons(self, texts: List[str], documents: List[Any], version: str) -> Dict[str, Any]:
        """分析比较"""
        results = {}
        
        # 按学科分组
        discipline_groups = self.reader.group_by_classification(documents, ClassificationField.DISCIPLINE)
        
        # 学科对比分析
        if len(discipline_groups) > 1:
            print(f"   学科对比分析...")
            metadata = self.reader.get_metadata_for_analysis(documents)
            comparator = DisciplineComparator(f"{version}_comparator", 
                                            self.analyzer_configs["discipline_comparator"])
            comparison_results = comparator.analyze(texts, metadata)
            
            if comparison_results and "error" not in comparison_results:
                stats = comparison_results["basic_statistics"]
                print(f"     分析学科数: {stats['total_disciplines']}")
                
                # 显示显著差异
                if "differences_analysis" in comparison_results:
                    diffs = comparison_results["differences_analysis"]
                    if "max_min_differences" in diffs:
                        for metric, info in diffs["max_min_differences"].items():
                            if info['relative_range_percent'] > 20:
                                print(f"     {metric}: {info['max_discipline']} 比 {info['min_discipline']} "
                                      f"高 {info['relative_range_percent']:.1f}%")
                                      
                results["discipline_comparison"] = comparison_results
        else:
            print(f"   跳过学科对比：只有 {len(discipline_groups)} 个学科")
            
        # 体裁对比分析
        print(f"   体裁对比分析...")
        
        # 准备体裁元数据
        genre_metadata = self._prepare_genre_metadata(documents)
        
        # 使用GenreComparator进行体裁对比分析
        genre_comparator = GenreComparator(f"{version}_genre_comparator", 
                                         self.analyzer_configs["genre_comparator"])
        genre_comparison_results = genre_comparator.analyze(texts, genre_metadata)
        
        if genre_comparison_results and "error" not in genre_comparison_results:
            stats = genre_comparison_results["basic_statistics"]
            print(f"     分析体裁数: {stats['total_genres']}")
            print(f"     体裁列表: {', '.join(stats['genres_analyzed'])}")
            
            # 显示体裁指标
            if "genre_metrics" in genre_comparison_results:
                genre_metrics = genre_comparison_results["genre_metrics"]
                for genre, metrics in genre_metrics.items():
                    print(f"     {genre}:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"       {metric}: {value:.4f}")
            
            # 显示显著差异
            if "differences_analysis" in genre_comparison_results:
                diffs = genre_comparison_results["differences_analysis"]
                if "max_min_differences" in diffs:
                    for metric, info in diffs["max_min_differences"].items():
                        if info['relative_range_percent'] > 20:
                            print(f"     {metric}: {info['max_genre']} 比 {info['min_genre']} "
                                  f"高 {info['relative_range_percent']:.1f}%")
            
            results["genre_comparison"] = genre_comparison_results
        else:
            print(f"     体裁对比分析失败: {genre_comparison_results.get('error', '未知错误')}")
            
        return results
        
    def _analyze_source_comparison(self, source_groups: Dict[str, List[Any]]) -> Dict[str, Any]:
        """分析源文件对比"""
        source_stats = {}
        
        for source, docs in source_groups.items():
            if len(docs) >= 3:  # 至少需要3个文档
                source_name = Path(source).stem
                lengths = [doc.text_length for doc in docs]
                word_counts = [doc.word_count for doc in docs]
                
                source_stats[source_name] = {
                    "count": len(docs),
                    "mean_length": np.mean(lengths),
                    "std_length": np.std(lengths),
                    "min_length": np.min(lengths),
                    "max_length": np.max(lengths),
                    "mean_words": np.mean(word_counts),
                    "std_words": np.std(word_counts)
                }
                
        # 显示差异最大的源文件
        if source_stats:
            sorted_stats = sorted(source_stats.items(), 
                                 key=lambda x: x[1]["mean_length"], 
                                 reverse=True)
            
            if len(sorted_stats) >= 2:
                longest = sorted_stats[0]
                shortest = sorted_stats[-1]
                length_diff = longest[1]["mean_length"] - shortest[1]["mean_length"]
                diff_percent = (length_diff / shortest[1]["mean_length"]) * 100
                
                print(f"     文本长度差异最大的体裁:")
                print(f"       最长: {longest[0]} (平均 {longest[1]['mean_length']:.0f} 字符)")
                print(f"       最短: {shortest[0]} (平均 {shortest[1]['mean_length']:.0f} 字符)")
                print(f"       差异: {length_diff:.0f} 字符 ({diff_percent:.1f}%)")
                
        return source_stats
        
    def _prepare_genre_metadata(self, documents: List[Any]) -> Dict[str, Any]:
        """准备体裁元数据"""
        genre_metadata = {
            "genre": [],
            "extraction_info": []
        }
        
        for doc in documents:
            # 从文档中提取体裁信息
            if hasattr(doc, 'genre') and doc.genre:
                genre_metadata["genre"].append(doc.genre)
            else:
                genre_metadata["genre"].append("unknown")
                
            # 从提取信息中获取体裁
            if hasattr(doc, 'extraction_info') and doc.extraction_info:
                extraction_info = doc.extraction_info
                if isinstance(extraction_info, dict) and "genre" in extraction_info:
                    genre_metadata["extraction_info"].append({
                        "genre": extraction_info.get("genre", "unknown"),
                        "source_field": extraction_info.get("source_field", ""),
                        "method": extraction_info.get("method", "")
                    })
                else:
                    genre_metadata["extraction_info"].append({"genre": "unknown"})
            else:
                genre_metadata["extraction_info"].append({"genre": "unknown"})
                
        return genre_metadata
        
    def create_visualizations(self, results: Dict[str, Any], documents: List[Any]):
        """创建可视化图表"""
        print(f"\n{'='*80}")
        print("创建可视化图表...")
        print(f"{'='*80}")
        
        # 1. 分类分布图
        self._create_classification_charts(results, documents)
        
        # 2. 文本特征图
        self._create_text_feature_charts(documents)
        
        # 3. 比较分析图
        self._create_comparison_charts(results, documents)
        
        print(f"\n可视化图表已保存到: {self.output_dir}")
        
    def _create_classification_charts(self, results: Dict[str, Any], documents: List[Any]):
        """创建分类分布图"""
        classification = results.get("classification", {})
        
        # 学科分布饼图
        if "disciplines" in classification:
            discipline_counts = classification["disciplines"]
            if discipline_counts:
                self._create_pie_chart(
                    data=discipline_counts,
                    title="Document Distribution by Discipline",
                    output_file=self.output_dir / "discipline_distribution.png",
                    color_palette=self.colors['discipline']
                )
                
        # 源文件分布饼图
        if "sources" in classification:
            source_counts = classification["sources"]
            if len(source_counts) <= 15:  # 如果数量不多，显示所有
                self._create_pie_chart(
                    data=source_counts,
                    title="Document Distribution by Source File (Genre)",
                    output_file=self.output_dir / "source_distribution.png",
                    color_palette=self.colors['genre']
                )
                
    def _create_text_feature_charts(self, documents: List[Any]):
        """创建文本特征图"""
        if not documents:
            return
            
        # 文本长度分布直方图
        text_lengths = [doc.text_length for doc in documents]
        self._create_histogram(
            data=text_lengths,
            title="Text Length Distribution",
            xlabel="Text Length (characters)",
            ylabel="Frequency",
            output_file=self.output_dir / "text_length_histogram.png",
            bins=30
        )
        
        # 词数分布直方图
        word_counts = [doc.word_count for doc in documents]
        self._create_histogram(
            data=word_counts,
            title="Word Count Distribution",
            xlabel="Word Count",
            ylabel="Frequency",
            output_file=self.output_dir / "word_count_histogram.png",
            bins=30
        )
        
        # 文本长度 vs 词数散点图
        self._create_scatter_plot(
            x_data=text_lengths,
            y_data=word_counts,
            title="Text Length vs Word Count",
            xlabel="Text Length (characters)",
            ylabel="Word Count",
            output_file=self.output_dir / "length_vs_words_scatter.png"
        )
        
    def _create_comparison_charts(self, results: Dict[str, Any], documents: List[Any]):
        """创建比较分析图"""
        # 学科间文本长度箱线图
        discipline_groups = self.reader.group_by_classification(documents, ClassificationField.DISCIPLINE)
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
                
        # 源文件间文本长度箱线图
        source_groups = self.reader.group_by_classification(documents, ClassificationField.SOURCE_FILE)
        if len(source_groups) > 1 and len(source_groups) <= 15:
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
                
        # 交叉分布热力图
        if len(discipline_groups) > 1 and len(source_groups) > 1:
            self._create_cross_distribution_heatmap(documents)
            
        # 8. 体裁与学科分组柱状图
        if "discipline_comparison" in results and "genre_comparison" in results:
            print(f"\n8. 创建体裁与学科分组柱状图...")
            self._create_genre_discipline_grouped_bar_charts(results, documents)
            
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
        
    def _create_cross_distribution_heatmap(self, documents: List[Any]):
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
        
    def _create_genre_discipline_grouped_bar_charts(self, results: Dict[str, Any], documents: List[Any]):
        """创建体裁与学科分组柱状图"""
        try:
            # 导入EnhancedVisualization
            from enhanced_visualization import EnhancedVisualization
            
            # 创建可视化器
            visualizer = EnhancedVisualization(str(self.output_dir))
            
            # 检查是否有必要的数据
            if "discipline_comparison" not in results or "genre_comparison" not in results:
                print("  警告: 缺少学科或体裁对比数据，跳过分组柱状图")
                return
            
            discipline_results = results["discipline_comparison"]
            genre_results = results["genre_comparison"]
            
            # 定义要显示的指标
            metrics = [
                "type_token_ratio",
                "mtld_average", 
                "academic_word_ratio",
                "discipline_word_ratio",
                "avg_word_length",
                "vocabulary_richness"
            ]
            
            # 创建综合图表
            summaries = visualizer.create_comprehensive_genre_discipline_chart(
                analysis_results=results,
                metrics=metrics
            )
            
            if summaries:
                print(f"  成功创建 {len(summaries)} 个分组柱状图")
                
                # 在报告中添加可视化文件列表
                self._add_grouped_bar_charts_to_report()
                
        except ImportError as e:
            print(f"  错误: 无法导入EnhancedVisualization - {e}")
        except Exception as e:
            print(f"  错误: 创建分组柱状图时发生错误 - {e}")
            import traceback
            traceback.print_exc()
            
    def _add_grouped_bar_charts_to_report(self):
        """在报告中添加分组柱状图文件列表"""
        # 查找所有分组柱状图文件
        grouped_bar_files = []
        for file_path in self.output_dir.glob("*grouped_bar*.png"):
            grouped_bar_files.append(file_path.name)
        
        for file_path in self.output_dir.glob("*metric_comparison*.png"):
            grouped_bar_files.append(file_path.name)
        
        if grouped_bar_files:
            # 更新Markdown报告
            report_file = self.output_dir / "analysis_report.md"
            if report_file.exists():
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 在可视化文件部分添加分组柱状图
                if "## 7. 生成的可视化文件" in content:
                    # 添加分组柱状图部分
                    grouped_bar_section = "\n### 分组柱状图（体裁与学科对比）\n\n"
                    for file_name in sorted(grouped_bar_files):
                        # 从文件名提取指标名称
                        if "type_token_ratio" in file_name:
                            metric_name = "TTR (词汇丰富度)"
                        elif "mtld_average" in file_name:
                            metric_name = "MTLD (文本词汇多样性)"
                        elif "academic_word_ratio" in file_name:
                            metric_name = "学术词汇比例"
                        elif "discipline_word_ratio" in file_name:
                            metric_name = "学科词汇比例"
                        elif "avg_word_length" in file_name:
                            metric_name = "平均词长"
                        elif "vocabulary_richness" in file_name:
                            metric_name = "词汇丰富度"
                        elif "metric_comparison" in file_name:
                            metric_name = "多指标对比"
                        else:
                            metric_name = "体裁与学科对比"
                        
                        grouped_bar_section += f"- `{file_name}` - {metric_name}分组柱状图\n"
                    
                    # 插入到可视化文件部分
                    insert_pos = content.find("## 7. 生成的可视化文件")
                    end_pos = content.find("\n## 8. 结论", insert_pos)
                    
                    if end_pos > insert_pos:
                        new_content = content[:end_pos] + grouped_bar_section + content[end_pos:]
                        
                        with open(report_file, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        print(f"  已更新报告文件，添加了 {len(grouped_bar_files)} 个分组柱状图")
        
    def generate_reports(self, results: Dict[str, Any], documents: List[Any]):
        """生成报告"""
        print(f"\n{'='*80}")
        print("生成分析报告...")
        print(f"{'='*80}")
        
        # 1. 生成JSON报告
        json_file = self.output_dir / "analysis_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"JSON报告已保存: {json_file}")
        
        # 2. 生成CSV报告
        csv_file = self.output_dir / "analysis_results.csv"
        self._generate_csv_report(results, csv_file)
        print(f"CSV报告已保存: {csv_file}")
        
        # 3. 生成Markdown报告
        md_file = self.output_dir / "analysis_report.md"
        self._generate_markdown_report(results, documents, md_file)
        print(f"Markdown报告已保存: {md_file}")
        
        print(f"\n所有报告已保存到: {self.output_dir}")
        
    def _generate_csv_report(self, results: Dict[str, Any], csv_file: Path):
        """生成CSV报告"""
        # 提取关键数据
        rows = []
        
        # 基本统计
        rows.append({
            "metric": "version",
            "value": results.get("version", ""),
            "category": "basic"
        })
        
        rows.append({
            "metric": "document_count",
            "value": results.get("document_count", 0),
            "category": "basic"
        })
        
        # 分类统计
        classification = results.get("classification", {})
        if "disciplines" in classification:
            for discipline, count in classification["disciplines"].items():
                rows.append({
                    "metric": f"discipline_{discipline}",
                    "value": count,
                    "category": "classification"
                })
                
        # 词汇分析
        vocab_results = results.get("vocabulary_analysis", {})
        if "basic_statistics" in vocab_results:
            stats = vocab_results["basic_statistics"]
            rows.append({
                "metric": "total_tokens",
                "value": stats.get("total_tokens", 0),
                "category": "vocabulary"
            })
            rows.append({
                "metric": "total_types",
                "value": stats.get("total_types", 0),
                "category": "vocabulary"
            })
            rows.append({
                "metric": "type_token_ratio",
                "value": stats.get("type_token_ratio", 0),
                "category": "vocabulary"
            })
            
        # 写入CSV
        if rows:
            fieldnames = ["metric", "value", "category"]
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
    def _generate_markdown_report(self, results: Dict[str, Any], documents: List[Any], md_file: Path):
        """生成Markdown报告"""
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# 体裁与学科分析报告\n\n")
            f.write(f"*生成时间: {results.get('timestamp', '')}*\n")
            f.write(f"*输出目录: {results.get('output_dir', '')}*\n\n")
            
            f.write(f"## 1. 概述\n\n")
            f.write(f"**分析版本:** {results.get('version', 'Unknown')}\n")
            f.write(f"**分析文档数:** {results.get('document_count', 0):,}\n\n")
            
            # 分类分析
            f.write("## 2. 分类分析\n\n")
            
            classification = results.get("classification", {})
            if "disciplines" in classification:
                f.write("### 2.1 学科分类\n\n")
                f.write("| 学科 | 文档数 | 百分比 |\n")
                f.write("|------|--------|--------|\n")
                
                discipline_counts = classification["disciplines"]
                total = sum(discipline_counts.values())
                
                for discipline, count in discipline_counts.items():
                    percentage = count / total * 100
                    f.write(f"| {discipline} | {count:,} | {percentage:.1f}% |\n")
                f.write("\n")
                
            if "sources" in classification:
                f.write("### 2.2 体裁分类（源文件）\n\n")
                f.write(f"**总源文件数:** {classification.get('total_sources', 0)}\n\n")
                
                source_counts = classification["sources"]
                sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:15]
                
                f.write("| 源文件（体裁） | 文档数 | 百分比 |\n")
                f.write("|---------------|--------|--------|\n")
                
                for source, count in sorted_sources:
                    percentage = count / total * 100
                    f.write(f"| {source} | {count:,} | {percentage:.1f}% |\n")
                f.write("\n")
                
            # 词汇分析
            f.write("## 3. 词汇分析\n\n")
            
            vocab_results = results.get("vocabulary_analysis", {})
            if "basic_statistics" in vocab_results:
                stats = vocab_results["basic_statistics"]
                
                f.write("### 3.1 基本统计\n\n")
                f.write(f"- **总词数:** {stats.get('total_tokens', 0):,}\n")
                f.write(f"- **总词型:** {stats.get('total_types', 0):,}\n")
                f.write(f"- **TTR (词汇丰富度):** {stats.get('type_token_ratio', 0):.4f}\n")
                f.write(f"- **平均词数/文本:** {stats.get('average_tokens_per_text', 0):.1f}\n")
                f.write(f"- **平均词型/文本:** {stats.get('average_types_per_text', 0):.1f}\n\n")
                
            # 学术词汇分析
            f.write("## 4. 学术词汇分析\n\n")
            
            academic_results = results.get("academic_analysis", {})
            if "academic_analysis" in academic_results:
                academic = academic_results["academic_analysis"]
                
                f.write("### 4.1 学术词汇比例\n\n")
                f.write(f"- **总体学术词汇比例:** {academic.get('overall_ratio', 0)*100:.2f}%\n")
                f.write(f"- **学科词汇比例:** {academic.get('discipline_ratio', 0)*100:.2f}%\n")
                f.write(f"- **交叉学科词汇比例:** {academic.get('cross_discipline_ratio', 0)*100:.2f}%\n\n")
                
            # 比较分析
            f.write("## 5. 比较分析\n\n")
            
            # 学科比较
            if "discipline_comparison" in results:
                f.write("### 5.1 学科间差异\n\n")
                comp = results["discipline_comparison"]
                if "differences_analysis" in comp:
                    diffs = comp["differences_analysis"]
                    if "max_min_differences" in diffs:
                        f.write("| 指标 | 最高学科 | 最低学科 | 相对差异 |\n")
                        f.write("|------|----------|----------|----------|\n")
                        
                        for metric, info in diffs["max_min_differences"].items():
                            if info['relative_range_percent'] > 20:
                                f.write(f"| {metric} | {info['max_discipline']} | {info['min_discipline']} | "
                                       f"{info['relative_range_percent']:.1f}% |\n")
                        f.write("\n")
                        
            # 体裁比较
            if "genre_comparison" in results:
                f.write("### 5.2 体裁间差异\n\n")
                genre_comparison = results["genre_comparison"]
                
                # 显示体裁基本信息
                if "basic_statistics" in genre_comparison:
                    stats = genre_comparison["basic_statistics"]
                    f.write(f"**分析体裁数:** {stats.get('total_genres', 0)}\n")
                    f.write(f"**体裁列表:** {', '.join(stats.get('genres_analyzed', []))}\n")
                    f.write(f"**总文本数:** {stats.get('total_texts', 0):,}\n\n")
                    
                    # 显示各体裁文本数
                    if "texts_by_genre" in stats:
                        f.write("| 体裁 | 文档数 | 百分比 |\n")
                        f.write("|------|--------|--------|\n")
                        
                        texts_by_genre = stats["texts_by_genre"]
                        total_texts = sum(texts_by_genre.values())
                        
                        for genre, count in texts_by_genre.items():
                            percentage = count / total_texts * 100 if total_texts > 0 else 0
                            f.write(f"| {genre} | {count:,} | {percentage:.1f}% |\n")
                        f.write("\n")
                
                # 显示体裁指标差异
                if "differences_analysis" in genre_comparison:
                    diffs = genre_comparison["differences_analysis"]
                    if "max_min_differences" in diffs:
                        f.write("**最大差异指标:**\n\n")
                        f.write("| 指标 | 最高体裁 | 最低体裁 | 相对差异 |\n")
                        f.write("|------|----------|----------|----------|\n")
                        
                        for metric, info in diffs["max_min_differences"].items():
                            if info.get('relative_range_percent', 0) > 20:
                                f.write(f"| {metric} | {info.get('max_genre', '')} | {info.get('min_genre', '')} | "
                                       f"{info.get('relative_range_percent', 0):.1f}% |\n")
                        f.write("\n")
                        
                # 显示体裁指标
                if "genre_metrics" in genre_comparison:
                    f.write("**体裁词汇指标:**\n\n")
                    genre_metrics = genre_comparison["genre_metrics"]
                    
                    for genre, metrics in genre_metrics.items():
                        f.write(f"**{genre}:**\n")
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                f.write(f"- {metric}: {value:.4f}\n")
                        f.write("\n")
                
            # Unknown语料详细信息
            if "unknown_documents_details" in results:
                f.write("## 6. Unknown语料详细信息\n\n")
                f.write(f"本次分析发现了 {len(results['unknown_documents_details'])} 个无法自动分类的语料（标记为Unknown）：\n\n")
                
                f.write("| 序号 | 文档ID | 源文件 | 文件名 | 文本长度 | 词数 | 标题 |\n")
                f.write("|------|--------|--------|--------|----------|------|------|\n")
                
                for i, doc_info in enumerate(results["unknown_documents_details"][:20], 1):
                    title = doc_info.get("title", "")
                    if len(title) > 30:
                        title = title[:27] + "..."
                    f.write(f"| {i} | {doc_info.get('id', '')} | {doc_info.get('source_file', '')} | "
                           f"{doc_info.get('file_name', '')} | {doc_info.get('text_length', 0):,} | "
                           f"{doc_info.get('word_count', 0):,} | {title} |\n")
                
                if len(results["unknown_documents_details"]) > 20:
                    f.write(f"\n*注：还有 {len(results['unknown_documents_details']) - 20} 个Unknown语料未在此表中显示*\n")
                
                f.write("\n**建议**：检查这些文件的文件名或内容，确保它们包含可识别的学科关键词。\n\n")
            
            # 可视化文件
            f.write("## 7. 生成的可视化文件\n\n")
            f.write("以下可视化文件已生成：\n\n")
            
            viz_files = [
                "discipline_distribution.png - 学科分布饼图",
                "source_distribution.png - 源文件（体裁）分布饼图",
                "text_length_histogram.png - 文本长度分布直方图",
                "word_count_histogram.png - 词数分布直方图",
                "length_vs_words_scatter.png - 文本长度与词数散点图",
                "discipline_length_boxplot.png - 学科间文本长度箱线图",
                "source_length_boxplot.png - 源文件间文本长度箱线图",
                "cross_distribution_heatmap.png - 学科与源文件交叉分布热力图"
            ]
            
            for viz in viz_files:
                f.write(f"- `{viz}`\n")
            
            f.write("\n## 8. 结论\n\n")
            f.write("本分析系统成功实现了对语料库的体裁和学科双重分类分析。主要发现包括：\n\n")
            f.write("1. **文档分布特征**：揭示了不同学科和体裁的文档分布模式\n")
            f.write("2. **文本特征差异**：展示了不同分类下的文本长度、词汇丰富度等特征差异\n")
            f.write("3. **学术词汇使用**：分析了学术词汇在不同分类中的使用情况\n")
            f.write("4. **比较分析**：提供了学科间和体裁间的量化比较\n")
            f.write("5. **Unknown语料识别**：识别并报告了无法自动分类的语料\n")
            f.write("6. **可视化呈现**：通过多种图表直观展示分析结果\n\n")
            
            f.write("这些分析结果为理解语料库的结构特征和内容特点提供了重要参考。\n")
            
    def run_complete_analysis(self, max_documents: int = 200):
        """
        运行完整的分析流程
        
        Args:
            max_documents: 最大文档数
        """
        print(f"\n{'='*80}")
        print("开始完整分析流程")
        print(f"{'='*80}")
        
        # 1. 分析数据
        results, documents = self.analyze_last_version(max_documents)
        
        if not results:
            print("分析失败，无法继续")
            return
            
        # 2. 创建可视化
        self.create_visualizations(results, documents)
        
        # 3. 生成报告
        self.generate_reports(results, documents)
        
        print(f"\n{'='*80}")
        print("分析完成!")
        print(f"{'='*80}")
        print(f"所有结果已保存到: {self.output_dir}")
        print(f"1. 分析报告: analysis_report.md")
        print(f"2. 数据文件: analysis_results.json, analysis_results.csv")
        print(f"3. 可视化图表: 8种不同类型的图表")
        print(f"{'='*80}")


def main():
    """主函数"""
    print("整合版体裁与学科分析系统")
    print("=" * 80)
    
    # 创建分析系统
    analyzer = IntegratedGenreDisciplineAnalysis("corpus")
    
    # 运行完整分析
    try:
        analyzer.run_complete_analysis(max_documents=1e15)
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n系统功能总结:")
    print("1. 增强版语料库读取器 - 支持完整元数据提取和嵌套结构识别")
    print("2. 体裁与学科双重分类 - 支持按源文件（体裁）和学科分类")
    print("3. 词汇分析 - 计算TTR、MTLD、词频等基本指标")
    print("4. 学术词汇分析 - 分析学术词汇使用情况")
    print("5. 比较分析 - 比较不同学科和体裁的特征差异")
    print("6. 可视化分析 - 生成多种图表展示分析结果")
    print("7. 报告生成 - 自动生成详细的分析报告")
    print("8. 时间戳输出 - 所有输出保存到以时间命名的文件夹中")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # 减少日志输出
    
    main()
