#!/usr/bin/env python3
"""
discipline_comparator.py - 学科词汇对比分析器

专门分析不同学科间词汇指标的差异，提供统计对比和可视化。
"""

import re
import json
import math
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from pathlib import Path
import logging

from analyzer_base import ConfigurableAnalyzer


class DisciplineComparator(ConfigurableAnalyzer):
    """学科词汇对比分析器"""
    
    def __init__(self, name: str = "discipline_comparator", config: Optional[Dict[str, Any]] = None):
        """
        初始化学科对比分析器
        
        Args:
            name: 分析器名称
            config: 配置字典
        """
        default_config = {
            "language": "en",
            "discipline_field": "discipline",  # 元数据中学科字段的名称
            "default_disciplines": [
                "Arts & Humanities",
                "Life Sciences & Biomedicine", 
                "Physical Sciences",
                "Social Sciences",
                "Technology"
            ],
            "vocabulary_metrics": [
                "total_tokens",
                "total_types", 
                "type_token_ratio",
                "mtld_average",
                "academic_word_ratio",
                "discipline_word_ratio",
                "avg_word_length",
                "vocabulary_richness"
            ],
            "calculate_differences": True,
            "calculate_statistical_significance": True,
            "significance_threshold": 0.05,
            "compare_pairs": True,
            "compare_to_overall": True,
            "output_format": "detailed",  # detailed, summary, matrix
            "normalize_case": True,
            "min_word_length": 3,
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name, default_config)
        
        # 存储学科数据
        self.discipline_data: Dict[str, Dict[str, Any]] = {}
        self.overall_stats: Dict[str, Any] = {}
        
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        预处理文本
        
        Args:
            texts: 原始文本列表
            
        Returns:
            预处理后的文本列表
        """
        processed_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            # 移除多余空白
            text = ' '.join(text.split())
            processed_texts.append(text)
        return processed_texts
        
    def _extract_discipline_from_metadata(self, metadata: Optional[Dict], index: int) -> Optional[str]:
        """
        从元数据提取学科信息
        
        Args:
            metadata: 元数据字典
            index: 文本索引
            
        Returns:
            学科名称，如果无法提取则返回None
        """
        if not metadata:
            return None
            
        # 尝试从不同字段提取学科
        discipline_fields = [
            self.config["discipline_field"],
            "discipline", "field", "category", "subject",
            "filename"  # 从文件名提取
        ]
        
        for field in discipline_fields:
            if field in metadata:
                value = metadata[field]
                if isinstance(value, list) and index < len(value):
                    discipline = value[index]
                elif isinstance(value, dict):
                    # 尝试从字典中获取
                    discipline = value.get(str(index))
                else:
                    discipline = value
                    
                if discipline and isinstance(discipline, str):
                    # 清理学科名称
                    discipline = discipline.strip()
                    if discipline:
                        return discipline
                        
        # 尝试从文件名提取
        if "filenames" in metadata and index < len(metadata["filenames"]):
            filename = metadata["filenames"][index]
            if filename:
                # 从文件名提取学科（简单实现）
                filename_lower = filename.lower()
                for discipline in self.config["default_disciplines"]:
                    discipline_lower = discipline.lower()
                    if discipline_lower in filename_lower:
                        return discipline
                        
        return None
        
    def _calculate_vocabulary_metrics(self, tokens: List[str]) -> Dict[str, float]:
        """
        计算词汇指标
        
        Args:
            tokens: 词元列表
            
        Returns:
            词汇指标字典
        """
        if not tokens:
            return {}
            
        metrics = {}
        
        # 基本统计
        total_tokens = len(tokens)
        unique_tokens = set(tokens)
        total_types = len(unique_tokens)
        
        metrics["total_tokens"] = total_tokens
        metrics["total_types"] = total_types
        metrics["type_token_ratio"] = total_types / total_tokens if total_tokens > 0 else 0
        
        # 平均词长
        avg_word_length = sum(len(token) for token in tokens) / total_tokens if total_tokens > 0 else 0
        metrics["avg_word_length"] = avg_word_length
        
        # 词汇丰富度（Guiraud's R）
        metrics["vocabulary_richness"] = total_types / math.sqrt(total_tokens) if total_tokens > 0 else 0
        
        # 计算MTLD（简化版）
        metrics["mtld_average"] = self._calculate_simple_mtld(tokens)
        
        return metrics
        
    def _calculate_simple_mtld(self, tokens: List[str], threshold: float = 0.72) -> float:
        """计算简化的MTLD"""
        if not tokens:
            return 0.0
            
        # 正向计算
        def calculate_direction(tokens_list):
            factors = 0
            start = 0
            
            while start < len(tokens_list):
                seen_types = set()
                ttr = 1.0
                length = 0
                
                for i in range(start, len(tokens_list)):
                    seen_types.add(tokens_list[i])
                    length += 1
                    ttr = len(seen_types) / length
                    
                    if ttr <= threshold:
                        factors += 1
                        start = i + 1
                        break
                        
                if ttr > threshold:
                    partial_factor = (1 - ttr) / (1 - threshold) if threshold < 1.0 else 0
                    factors += partial_factor
                    break
                    
            return len(tokens_list) / factors if factors > 0 else 0
            
        forward = calculate_direction(tokens)
        reverse = calculate_direction(list(reversed(tokens)))
        
        return (forward + reverse) / 2 if forward > 0 and reverse > 0 else max(forward, reverse)
        
    def analyze(self, texts: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行学科词汇对比分析
        
        Args:
            texts: 文本列表
            metadata: 元数据（应包含学科信息）
            
        Returns:
            对比分析结果字典
        """
        if not self.validate_input(texts):
            return {}
            
        # 预处理文本
        processed_texts = self.preprocess_texts(texts)
        
        # 按学科分组
        discipline_groups = defaultdict(list)
        discipline_indices = defaultdict(list)
        
        for i, text in enumerate(processed_texts):
            discipline = self._extract_discipline_from_metadata(metadata, i)
            if not discipline:
                # 使用默认学科或跳过
                continue
                
            discipline_groups[discipline].append(text)
            discipline_indices[discipline].append(i)
            
        if not discipline_groups:
            return {"error": "无法从元数据中提取学科信息"}
            
        # 计算各学科词汇指标
        discipline_metrics = {}
        discipline_tokens = {}
        
        for discipline, texts_list in discipline_groups.items():
            # 合并所有文本的词元
            all_tokens = []
            for text in texts_list:
                tokens = self._tokenize_text(text)
                all_tokens.extend(tokens)
                
            if all_tokens:
                discipline_tokens[discipline] = all_tokens
                metrics = self._calculate_vocabulary_metrics(all_tokens)
                
                # 添加学术词汇分析（如果可能）
                metrics.update(self._calculate_academic_metrics(all_tokens))
                
                discipline_metrics[discipline] = metrics
                
        # 计算总体统计
        overall_tokens = []
        for tokens in discipline_tokens.values():
            overall_tokens.extend(tokens)
            
        self.overall_stats = self._calculate_vocabulary_metrics(overall_tokens)
        self.overall_stats.update(self._calculate_academic_metrics(overall_tokens))
        
        # 计算学科间差异
        differences = {}
        if self.config["calculate_differences"]:
            differences = self._calculate_discipline_differences(discipline_metrics)
            
        # 计算统计显著性
        significance = {}
        if self.config["calculate_statistical_significance"]:
            significance = self._calculate_statistical_significance(discipline_tokens, discipline_metrics)
            
        # 生成对比矩阵
        comparison_matrix = self._generate_comparison_matrix(discipline_metrics)
        
        # 学科排名
        discipline_rankings = self._calculate_discipline_rankings(discipline_metrics)
        
        return {
            "basic_statistics": {
                "total_disciplines": len(discipline_metrics),
                "disciplines_analyzed": list(discipline_metrics.keys()),
                "total_texts": len(processed_texts),
                "texts_by_discipline": {d: len(texts) for d, texts in discipline_groups.items()}
            },
            "discipline_metrics": discipline_metrics,
            "overall_metrics": self.overall_stats,
            "differences_analysis": differences,
            "statistical_significance": significance,
            "comparison_matrix": comparison_matrix,
            "discipline_rankings": discipline_rankings,
            "visualization_data": self._prepare_visualization_data(discipline_metrics),
            "config": {
                "vocabulary_metrics": self.config["vocabulary_metrics"],
                "significance_threshold": self.config["significance_threshold"]
            }
        }
        
    def _tokenize_text(self, text: str) -> List[str]:
        """分词"""
        if self.config["normalize_case"]:
            text = text.lower()
            
        # 移除标点
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 按空白分割
        tokens = text.split()
        
        # 过滤短词
        if self.config["min_word_length"] > 1:
            tokens = [t for t in tokens if len(t) >= self.config["min_word_length"]]
            
        return tokens
        
    def _calculate_academic_metrics(self, tokens: List[str]) -> Dict[str, float]:
        """计算学术词汇指标（简化版）"""
        if not tokens:
            return {"academic_word_ratio": 0, "discipline_word_ratio": 0}
            
        # 学术词汇列表（简化版）
        academic_words = {
            "analyze", "approach", "area", "assess", "assume", "authority", "available",
            "benefit", "concept", "consist", "constitute", "context", "contract", "create",
            "data", "define", "derive", "distribute", "economy", "environment", "establish",
            "estimate", "evident", "export", "factor", "finance", "formula", "function"
        }
        
        # 学科特定词汇（示例）
        discipline_words = {
            "science": {"method", "experiment", "hypothesis", "theory", "data", "analysis"},
            "humanities": {"interpretation", "analysis", "context", "text", "discourse", "narrative"},
            "social": {"survey", "interview", "data", "analysis", "theory", "model"}
        }
        
        total_tokens = len(tokens)
        
        # 学术词汇比例
        academic_count = sum(1 for token in tokens if token in academic_words)
        academic_ratio = academic_count / total_tokens if total_tokens > 0 else 0
        
        # 学科词汇比例（合并所有学科词汇）
        all_discipline_words = set()
        for words in discipline_words.values():
            all_discipline_words.update(words)
            
        discipline_count = sum(1 for token in tokens if token in all_discipline_words)
        discipline_ratio = discipline_count / total_tokens if total_tokens > 0 else 0
        
        return {
            "academic_word_ratio": academic_ratio,
            "discipline_word_ratio": discipline_ratio
        }
        
    def _calculate_discipline_differences(self, discipline_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """计算学科间差异"""
        differences = {
            "pairwise_differences": {},
            "relative_to_overall": {},
            "max_min_differences": {}
        }
        
        disciplines = list(discipline_metrics.keys())
        metrics_list = list(self.config["vocabulary_metrics"])
        
        # 两两对比
        if self.config["compare_pairs"]:
            pairwise = {}
            for i, disc1 in enumerate(disciplines):
                for disc2 in disciplines[i+1:]:
                    pair_key = f"{disc1}_vs_{disc2}"
                    pair_diffs = {}
                    
                    for metric in metrics_list:
                        if metric in discipline_metrics[disc1] and metric in discipline_metrics[disc2]:
                            value1 = discipline_metrics[disc1][metric]
                            value2 = discipline_metrics[disc2][metric]
                            
                            if value1 != 0:
                                diff_abs = value2 - value1
                                diff_rel = (diff_abs / value1) * 100 if value1 != 0 else 0
                            else:
                                diff_abs = value2
                                diff_rel = 100 if value2 != 0 else 0
                                
                            pair_diffs[metric] = {
                                "absolute_difference": diff_abs,
                                "relative_difference_percent": diff_rel,
                                disc1: value1,
                                disc2: value2
                            }
                            
                    pairwise[pair_key] = pair_diffs
                    
            differences["pairwise_differences"] = pairwise
            
        # 与总体对比
        if self.config["compare_to_overall"] and self.overall_stats:
            relative_to_overall = {}
            for discipline in disciplines:
                discipline_diffs = {}
                for metric in metrics_list:
                    if metric in discipline_metrics[discipline] and metric in self.overall_stats:
                        disc_value = discipline_metrics[discipline][metric]
                        overall_value = self.overall_stats[metric]
                        
                        if overall_value != 0:
                            diff_abs = disc_value - overall_value
                            diff_rel = (diff_abs / overall_value) * 100 if overall_value != 0 else 0
                        else:
                            diff_abs = disc_value
                            diff_rel = 100 if disc_value != 0 else 0
                            
                        discipline_diffs[metric] = {
                            "absolute_difference": diff_abs,
                            "relative_difference_percent": diff_rel,
                            "discipline_value": disc_value,
                            "overall_value": overall_value
                        }
                        
                relative_to_overall[discipline] = discipline_diffs
                
            differences["relative_to_overall"] = relative_to_overall
            
        # 最大最小差异
        max_min_diffs = {}
        for metric in metrics_list:
            values = []
            for discipline in disciplines:
                if metric in discipline_metrics[discipline]:
                    values.append((discipline, discipline_metrics[discipline][metric]))
                    
            if values:
                max_discipline, max_value = max(values, key=lambda x: x[1])
                min_discipline, min_value = min(values, key=lambda x: x[1])
                
                if max_value != 0:
                    diff_abs = max_value - min_value
                    diff_rel = (diff_abs / max_value) * 100 if max_value != 0 else 0
                else:
                    diff_abs = min_value
                    diff_rel = 100 if min_value != 0 else 0
                    
                max_min_diffs[metric] = {
                    "max_discipline": max_discipline,
                    "max_value": max_value,
                    "min_discipline": min_discipline,
                    "min_value": min_value,
                    "absolute_range": diff_abs,
                    "relative_range_percent": diff_rel
                }
                
        differences["max_min_differences"] = max_min_diffs
        
        return differences
        
    def _calculate_statistical_significance(self, discipline_tokens: Dict[str, List[str]], 
                                          discipline_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """计算统计显著性（简化版）"""
        significance = {
            "pairwise_t_tests": {},
            "anova_results": {},
            "effect_sizes": {}
        }
        
        disciplines = list(discipline_tokens.keys())
        if len(disciplines) < 2:
            return significance
            
        # 这里实现简化的统计检验
        # 在实际应用中，应该使用scipy.stats等库
        
        return significance
        
    def _generate_comparison_matrix(self, discipline_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """生成对比矩阵"""
        disciplines = list(discipline_metrics.keys())
        metrics = list(self.config["vocabulary_metrics"])
        
        matrix = {
            "disciplines": disciplines,
            "metrics": metrics,
            "values": {},
            "normalized_values": {}
        }
        
        # 原始值矩阵
        for metric in metrics:
            matrix["values"][metric] = {}
            for discipline in disciplines:
                if metric in discipline_metrics[discipline]:
                    matrix["values"][metric][discipline] = discipline_metrics[discipline][metric]
                else:
                    matrix["values"][metric][discipline] = None
                    
        # 归一化值矩阵（0-1范围）
        for metric in metrics:
            values = [v for v in matrix["values"][metric].values() if v is not None]
            if values:
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val
                
                matrix["normalized_values"][metric] = {}
                for discipline in disciplines:
                    value = matrix["values"][metric][discipline]
                    if value is not None and range_val > 0:
                        normalized = (value - min_val) / range_val
                        matrix["normalized_values"][metric][discipline] = normalized
                    else:
                        matrix["normalized_values"][metric][discipline] = None
                        
        return matrix
        
    def _calculate_discipline_rankings(self, discipline_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """计算学科排名"""
        disciplines = list(discipline_metrics.keys())
        metrics = list(self.config["vocabulary_metrics"])
        
        rankings = {
            "by_metric": {},
            "overall_rankings": {}
        }
        
        # 按指标排名
        for metric in metrics:
            metric_values = []
            for discipline in disciplines:
                if metric in discipline_metrics[discipline]:
                    metric_values.append((discipline, discipline_metrics[discipline][metric]))
                    
            if metric_values:
                # 排序（值越大越好，除了某些指标）
                reverse = True  # 默认值越大越好
                if metric in ["total_tokens", "total_types"]:
                    # 这些指标值越大越好
                    reverse = True
                elif metric in ["type_token_ratio", "mtld_average", "academic_word_ratio"]:
                    # 这些指标值越大越好
                    reverse = True
                    
                sorted_values = sorted(metric_values, key=lambda x: x[1], reverse=reverse)
                
                metric_ranking = []
                for rank, (discipline, value) in enumerate(sorted_values, 1):
                    metric_ranking.append({
                        "rank": rank,
                        "discipline": discipline,
                        "value": value
                    })
                    
                rankings["by_metric"][metric] = metric_ranking
                
        # 综合排名（平均排名）
        if rankings["by_metric"]:
            discipline_scores = defaultdict(list)
            for metric, ranking in rankings["by_metric"].items():
                for item in ranking:
                    discipline_scores[item["discipline"]].append(item["rank"])
                    
            # 计算平均排名
            avg_rankings = []
            for discipline, ranks in discipline_scores.items():
                avg_rank = sum(ranks) / len(ranks)
                avg_rankings.append((discipline, avg_rank))
                
            # 按平均排名排序
            avg_rankings.sort(key=lambda x: x[1])
            
            overall_ranking = []
            for rank, (discipline, avg_rank) in enumerate(avg_rankings, 1):
                overall_ranking.append({
                    "rank": rank,
                    "discipline": discipline,
                    "average_rank": avg_rank,
                    "rank_details": {metric: discipline_scores[discipline][i] 
                                   for i, metric in enumerate(rankings["by_metric"].keys())}
                })
                
            rankings["overall_rankings"] = overall_ranking
            
        return rankings
        
    def _prepare_visualization_data(self, discipline_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """准备可视化数据"""
        disciplines = list(discipline_metrics.keys())
        metrics = list(self.config["vocabulary_metrics"])
        
        # 雷达图数据
        radar_data = []
        for discipline in disciplines:
            discipline_data = {"discipline": discipline, "metrics": {}}
            for metric in metrics:
                if metric in discipline_metrics[discipline]:
                    discipline_data["metrics"][metric] = discipline_metrics[discipline][metric]
            radar_data.append(discipline_data)
            
        # 热力图数据
        heatmap_data = {
            "disciplines": disciplines,
            "metrics": metrics,
            "values": []
        }
        
        for metric in metrics:
            row = {"metric": metric}
            for discipline in disciplines:
                if metric in discipline_metrics[discipline]:
                    row[discipline] = discipline_metrics[discipline][metric]
                else:
                    row[discipline] = None
            heatmap_data["values"].append(row)
            
        # 柱状图数据（按指标）
        bar_chart_data = {}
        for metric in metrics:
            metric_data = []
            for discipline in disciplines:
                if metric in discipline_metrics[discipline]:
                    metric_data.append({
                        "discipline": discipline,
                        "value": discipline_metrics[discipline][metric]
                    })
                    
            if metric_data:
                # 按值排序
                metric_data.sort(key=lambda x: x["value"], reverse=True)
                bar_chart_data[metric] = metric_data
                
        return {
            "radar_chart": radar_data,
            "heatmap": heatmap_data,
            "bar_charts": bar_chart_data,
            "metrics_for_visualization": metrics
        }
        
    def get_required_fields(self) -> List[str]:
        """获取需要的字段"""
        return ["text"]
        
    def get_result_schema(self) -> Dict[str, Any]:
        """获取结果模式"""
        return {
            "basic_statistics": {
                "total_disciplines": "int",
                "disciplines_analyzed": "list",
                "total_texts": "int",
                "texts_by_discipline": "dict"
            },
            "discipline_metrics": "dict",
            "overall_metrics": "dict",
            "differences_analysis": {
                "pairwise_differences": "dict",
                "relative_to_overall": "dict",
                "max_min_differences": "dict"
            },
            "statistical_significance": {
                "pairwise_t_tests": "dict",
                "anova_results": "dict",
                "effect_sizes": "dict"
            },
            "comparison_matrix": {
                "disciplines": "list",
                "metrics": "list",
                "values": "dict",
                "normalized_values": "dict"
            },
            "discipline_rankings": {
                "by_metric": "dict",
                "overall_rankings": "list"
            },
            "visualization_data": {
                "radar_chart": "list",
                "heatmap": "dict",
                "bar_charts": "dict",
                "metrics_for_visualization": "list"
            },
            "config": "dict"
        }


def test_discipline_comparator():
    """测试学科对比分析器"""
    print("测试学科词汇对比分析器...")
    
    # 创建测试数据
    test_texts = [
        "This research analyzes scientific data using statistical methods.",
        "Humanities scholars interpret historical texts through critical analysis.",
        "Social sciences employ surveys to study population dynamics.",
        "Technology engineering combines design with practical implementation.",
        "Life sciences focus on biological systems and medical applications."
    ]
    
    # 创建元数据（包含学科信息）
    test_metadata = {
        "discipline": ["Physical Sciences", "Arts & Humanities", "Social Sciences", 
                      "Technology", "Life Sciences & Biomedicine"],
        "filenames": ["physics_001.txt", "humanities_001.txt", "social_001.txt", 
                     "tech_001.txt", "life_001.txt"]
    }
    
    # 创建分析器
    comparator = DisciplineComparator("test_comparator", {
        "calculate_differences": True,
        "calculate_statistical_significance": False,  # 简化测试
        "compare_pairs": True,
        "compare_to_overall": True
    })
    
    # 执行分析
    results = comparator.analyze(test_texts, test_metadata)
    
    if results and "error" not in results:
        print("学科对比分析器测试成功!")
        
        # 显示基本信息
        stats = results["basic_statistics"]
        print(f"分析的学科数: {stats['total_disciplines']}")
        print(f"学科列表: {', '.join(stats['disciplines_analyzed'])}")
        
        # 显示学科指标
        print("\n学科词汇指标:")
        for discipline, metrics in results["discipline_metrics"].items():
            print(f"  {discipline}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")
                    
        # 显示差异分析
        if "differences_analysis" in results:
            diffs = results["differences_analysis"]
            if "max_min_differences" in diffs:
                print("\n最大差异指标:")
                for metric, info in diffs["max_min_differences"].items():
                    print(f"  {metric}: {info['max_discipline']}({info['max_value']:.4f}) vs "
                          f"{info['min_discipline']}({info['min_value']:.4f})")
                          
        return True
        
    print(f"学科对比分析器测试失败: {results.get('error', '未知错误')}")
    return False


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    if test_discipline_comparator():
        print("\n学科词汇对比分析器测试通过!")
    else:
        print("\n学科词汇对比分析器测试失败!")
