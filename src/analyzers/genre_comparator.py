#!/usr/bin/env python3
"""
genre_comparator.py - 体裁词汇对比分析器

专门分析不同体裁间词汇指标的差异，提供统计对比和可视化。
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


class GenreComparator(ConfigurableAnalyzer):
    """体裁词汇对比分析器"""
    
    def __init__(self, name: str = "genre_comparator", config: Optional[Dict[str, Any]] = None):
        """
        初始化体裁对比分析器
        
        Args:
            name: 分析器名称
            config: 配置字典
        """
        default_config = {
            "language": "en",
            "genre_field": "genre",  # 元数据中体裁字段的名称
            "default_genres": [
                "science_news",
                "academic_paper"
            ],
            "vocabulary_metrics": [
                "total_tokens",
                "total_types", 
                "type_token_ratio",
                "mtld_average",
                "academic_word_ratio",
                "discipline_word_ratio",
                "avg_word_length",
                "vocabulary_richness",
                "avg_sentence_length",
                "sentence_complexity"
            ],
            "calculate_differences": True,
            "calculate_statistical_significance": True,
            "significance_threshold": 0.05,
            "compare_pairs": True,
            "compare_to_overall": True,
            "output_format": "detailed",  # detailed, summary, matrix
            "normalize_case": True,
            "min_word_length": 3,
            "sentence_delimiters": ['.', '!', '?', ';'],
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name, default_config)
        
        # 存储体裁数据
        self.genre_data: Dict[str, Dict[str, Any]] = {}
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
        
    def _extract_genre_from_metadata(self, metadata: Optional[Dict], index: int) -> Optional[str]:
        """
        从元数据提取体裁信息
        
        Args:
            metadata: 元数据字典
            index: 文本索引
            
        Returns:
            体裁名称，如果无法提取则返回None
        """
        if not metadata:
            return None
            
        # 尝试从不同字段提取体裁
        genre_fields = [
            self.config["genre_field"],
            "genre", "type", "category", "source_type",
            "extraction_info.genre"  # 从提取信息中获取
        ]
        
        for field in genre_fields:
            if field in metadata:
                value = metadata[field]
                if isinstance(value, list) and index < len(value):
                    genre = value[index]
                elif isinstance(value, dict):
                    # 尝试从字典中获取
                    genre = value.get(str(index))
                else:
                    genre = value
                    
                if genre and isinstance(genre, str):
                    # 清理体裁名称
                    genre = genre.strip()
                    if genre:
                        return genre
                        
        # 尝试从提取信息中获取
        if "extraction_info" in metadata and index < len(metadata["extraction_info"]):
            extraction_info = metadata["extraction_info"][index]
            if isinstance(extraction_info, dict) and "genre" in extraction_info:
                genre = extraction_info["genre"]
                if genre and isinstance(genre, str):
                    return genre
                    
        return None
        
    def _calculate_vocabulary_metrics(self, tokens: List[str], text: str = "") -> Dict[str, float]:
        """
        计算词汇指标
        
        Args:
            tokens: 词元列表
            text: 原始文本（用于句子分析）
            
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
        
        # 句子分析（如果有原始文本）
        if text:
            sentence_metrics = self._analyze_sentences(text)
            metrics.update(sentence_metrics)
        
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
    
    def _analyze_sentences(self, text: str) -> Dict[str, float]:
        """分析句子特征"""
        if not text:
            return {}
            
        # 分割句子
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in self.config["sentence_delimiters"]:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # 添加最后一个句子（如果没有以分隔符结尾）
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        if not sentences:
            return {}
            
        # 计算句子长度
        sentence_lengths = []
        for sentence in sentences:
            words = sentence.split()
            sentence_lengths.append(len(words))
        
        # 计算指标
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        # 句子复杂度（长句比例）
        long_sentence_threshold = 20  # 超过20词的句子被认为是长句
        long_sentence_count = sum(1 for length in sentence_lengths if length > long_sentence_threshold)
        sentence_complexity = long_sentence_count / len(sentence_lengths) if sentence_lengths else 0
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "sentence_complexity": sentence_complexity,
            "total_sentences": len(sentences)
        }
        
    def analyze(self, texts: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行体裁词汇对比分析
        
        Args:
            texts: 文本列表
            metadata: 元数据（应包含体裁信息）
            
        Returns:
            对比分析结果字典
        """
        if not self.validate_input(texts):
            return {}
            
        # 预处理文本
        processed_texts = self.preprocess_texts(texts)
        
        # 按体裁分组
        genre_groups = defaultdict(list)
        genre_indices = defaultdict(list)
        
        for i, text in enumerate(processed_texts):
            genre = self._extract_genre_from_metadata(metadata, i)
            if not genre:
                # 使用默认体裁或跳过
                continue
                
            genre_groups[genre].append(text)
            genre_indices[genre].append(i)
            
        if not genre_groups:
            return {"error": "无法从元数据中提取体裁信息"}
            
        # 计算各体裁词汇指标
        genre_metrics = {}
        genre_tokens = {}
        
        for genre, texts_list in genre_groups.items():
            # 合并所有文本的词元
            all_tokens = []
            for text in texts_list:
                tokens = self._tokenize_text(text)
                all_tokens.extend(tokens)
                
            if all_tokens:
                genre_tokens[genre] = all_tokens
                # 使用第一个文本进行句子分析
                sample_text = texts_list[0] if texts_list else ""
                metrics = self._calculate_vocabulary_metrics(all_tokens, sample_text)
                
                # 添加学术词汇分析（如果可能）
                metrics.update(self._calculate_academic_metrics(all_tokens))
                
                genre_metrics[genre] = metrics
                
        # 计算总体统计
        overall_tokens = []
        for tokens in genre_tokens.values():
            overall_tokens.extend(tokens)
            
        self.overall_stats = self._calculate_vocabulary_metrics(overall_tokens)
        self.overall_stats.update(self._calculate_academic_metrics(overall_tokens))
        
        # 计算体裁间差异
        differences = {}
        if self.config["calculate_differences"]:
            differences = self._calculate_genre_differences(genre_metrics)
            
        # 计算统计显著性
        significance = {}
        if self.config["calculate_statistical_significance"]:
            significance = self._calculate_statistical_significance(genre_tokens, genre_metrics)
            
        # 生成对比矩阵
        comparison_matrix = self._generate_comparison_matrix(genre_metrics)
        
        # 体裁排名
        genre_rankings = self._calculate_genre_rankings(genre_metrics)
        
        return {
            "basic_statistics": {
                "total_genres": len(genre_metrics),
                "genres_analyzed": list(genre_metrics.keys()),
                "total_texts": len(processed_texts),
                "texts_by_genre": {g: len(texts) for g, texts in genre_groups.items()}
            },
            "genre_metrics": genre_metrics,
            "overall_metrics": self.overall_stats,
            "differences_analysis": differences,
            "statistical_significance": significance,
            "comparison_matrix": comparison_matrix,
            "genre_rankings": genre_rankings,
            "visualization_data": self._prepare_visualization_data(genre_metrics),
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
            "estimate", "evident", "export", "factor", "finance", "formula", "function",
            "identify", "income", "indicate", "individual", "interpret", "involve", "issue",
            "labour", "legal", "legislate", "major", "method", "occur", "percent", "period",
            "policy", "principle", "proceed", "process", "require", "research", "respond",
            "role", "section", "sector", "significant", "similar", "source", "specific",
            "structure", "theory", "vary"
        }
        
        # 学科特定词汇（示例）
        discipline_words = {
            "science": {"method", "experiment", "hypothesis", "theory", "data", "analysis", "result", "conclusion"},
            "humanities": {"interpretation", "analysis", "context", "text", "discourse", "narrative", "culture", "history"},
            "social": {"survey", "interview", "data", "analysis", "theory", "model", "population", "society"}
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
        
    def _calculate_genre_differences(self, genre_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """计算体裁间差异"""
        differences = {
            "pairwise_differences": {},
            "relative_to_overall": {},
            "max_min_differences": {}
        }
        
        genres = list(genre_metrics.keys())
        metrics_list = list(self.config["vocabulary_metrics"])
        
        # 两两对比
        if self.config["compare_pairs"]:
            pairwise = {}
            for i, genre1 in enumerate(genres):
                for genre2 in genres[i+1:]:
                    pair_key = f"{genre1}_vs_{genre2}"
                    pair_diffs = {}
                    
                    for metric in metrics_list:
                        if metric in genre_metrics[genre1] and metric in genre_metrics[genre2]:
                            value1 = genre_metrics[genre1][metric]
                            value2 = genre_metrics[genre2][metric]
                            
                            if value1 != 0:
                                diff_abs = value2 - value1
                                diff_rel = (diff_abs / value1) * 100 if value1 != 0 else 0
                            else:
                                diff_abs = value2
                                diff_rel = 100 if value2 != 0 else 0
                                
                            pair_diffs[metric] = {
                                "absolute_difference": diff_abs,
                                "relative_difference_percent": diff_rel,
                                genre1: value1,
                                genre2: value2
                            }
                            
                    pairwise[pair_key] = pair_diffs
                    
            differences["pairwise_differences"] = pairwise
            
        # 与总体对比
        if self.config["compare_to_overall"] and self.overall_stats:
            relative_to_overall = {}
            for genre in genres:
                genre_diffs = {}
                for metric in metrics_list:
                    if metric in genre_metrics[genre] and metric in self.overall_stats:
                        genre_value = genre_metrics[genre][metric]
                        overall_value = self.overall_stats[metric]
                        
                        if overall_value != 0:
                            diff_abs = genre_value - overall_value
                            diff_rel = (diff_abs / overall_value) * 100 if overall_value != 0 else 0
                        else:
                            diff_abs = genre_value
                            diff_rel = 100 if genre_value != 0 else 0
                            
                        genre_diffs[metric] = {
                            "absolute_difference": diff_abs,
                            "relative_difference_percent": diff_rel,
                            "genre_value": genre_value,
                            "overall_value": overall_value
                        }
                        
                relative_to_overall[genre] = genre_diffs
                
            differences["relative_to_overall"] = relative_to_overall
            
        # 最大最小差异
        max_min_diffs = {}
        for metric in metrics_list:
            values = []
            for genre in genres:
                if metric in genre_metrics[genre]:
                    values.append((genre, genre_metrics[genre][metric]))
                    
            if values:
                max_genre, max_value = max(values, key=lambda x: x[1])
                min_genre, min_value = min(values, key=lambda x: x[1])
                
                if max_value != 0:
                    diff_abs = max_value - min_value
                    diff_rel = (diff_abs / max_value) * 100 if max_value != 0 else 0
                else:
                    diff_abs = min_value
                    diff_rel = 100 if min_value != 0 else 0
                    
                max_min_diffs[metric] = {
                    "max_genre": max_genre,
                    "max_value": max_value,
                    "min_genre": min_genre,
                    "min_value": min_value,
                    "absolute_range": diff_abs,
                    "relative_range_percent": diff_rel
                }
                
        differences["max_min_differences"] = max_min_diffs
        
        return differences
        
    def _calculate_statistical_significance(self, genre_tokens: Dict[str, List[str]], 
                                          genre_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """计算统计显著性（简化版）"""
        significance = {
            "pairwise_t_tests": {},
            "anova_results": {},
            "effect_sizes": {}
        }
        
        genres = list(genre_tokens.keys())
        if len(genres) < 2:
            return significance
            
        # 这里实现简化的统计检验
        # 在实际应用中，应该使用scipy.stats等库
        
        return significance
        
    def _generate_comparison_matrix(self, genre_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """生成对比矩阵"""
        genres = list(genre_metrics.keys())
        metrics = list(self.config["vocabulary_metrics"])
        
        matrix = {
            "genres": genres,
            "metrics": metrics,
            "values": {},
            "normalized_values": {}
        }
        
        # 原始值矩阵
        for metric in metrics:
            matrix["values"][metric] = {}
            for genre in genres:
                if metric in genre_metrics[genre]:
                    matrix["values"][metric][genre] = genre_metrics[genre][metric]
                else:
                    matrix["values"][metric][genre] = None
                    
        # 归一化值矩阵（0-1范围）
        for metric in metrics:
            values = [v for v in matrix["values"][metric].values() if v is not None]
            if values:
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val
                
                matrix["normalized_values"][metric] = {}
                for genre in genres:
                    value = matrix["values"][metric][genre]
                    if value is not None and range_val > 0:
                        normalized = (value - min_val) / range_val
                        matrix["normalized_values"][metric][genre] = normalized
                    else:
                        matrix["normalized_values"][metric][genre] = None
                        
        return matrix
        
    def _calculate_genre_rankings(self, genre_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """计算体裁排名"""
        genres = list(genre_metrics.keys())
        metrics = list(self.config["vocabulary_metrics"])
        
        rankings = {
            "by_metric": {},
            "overall_rankings": {}
        }
        
        # 按指标排名
        for metric in metrics:
            metric_values = []
            for genre in genres:
                if metric in genre_metrics[genre]:
                    metric_values.append((genre, genre_metrics[genre][metric]))
                    
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
                for rank, (genre, value) in enumerate(sorted_values, 1):
                    metric_ranking.append({
                        "rank": rank,
                        "genre": genre,
                        "value": value
                    })
                    
                rankings["by_metric"][metric] = metric_ranking
                
        # 综合排名（平均排名）
        if rankings["by_metric"]:
            genre_scores = defaultdict(list)
            for metric, ranking in rankings["by_metric"].items():
                for item in ranking:
                    genre_scores[item["genre"]].append(item["rank"])
                    
            # 计算平均排名
            avg_rankings = []
            for genre, ranks in genre_scores.items():
                avg_rank = sum(ranks) / len(ranks)
                avg_rankings.append((genre, avg_rank))
                
            # 按平均排名排序
            avg_rankings.sort(key=lambda x: x[1])
            
            overall_ranking = []
            for rank, (genre, avg_rank) in enumerate(avg_rankings, 1):
                overall_ranking.append({
                    "rank": rank,
                    "genre": genre,
                    "average_rank": avg_rank,
                    "rank_details": {metric: genre_scores[genre][i] 
                                   for i, metric in enumerate(rankings["by_metric"].keys())}
                })
                
            rankings["overall_rankings"] = overall_ranking
            
        return rankings
        
    def _prepare_visualization_data(self, genre_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """准备可视化数据"""
        genres = list(genre_metrics.keys())
        metrics = list(self.config["vocabulary_metrics"])
        
        # 雷达图数据
        radar_data = []
        for genre in genres:
            genre_data = {"genre": genre, "metrics": {}}
            for metric in metrics:
                if metric in genre_metrics[genre]:
                    genre_data["metrics"][metric] = genre_metrics[genre][metric]
            radar_data.append(genre_data)
            
        # 热力图数据
        heatmap_data = {
            "genres": genres,
            "metrics": metrics,
            "values": []
        }
        
        for metric in metrics:
            row = {"metric": metric}
            for genre in genres:
                if metric in genre_metrics[genre]:
                    row[genre] = genre_metrics[genre][metric]
                else:
                    row[genre] = None
            heatmap_data["values"].append(row)
            
        # 柱状图数据（按指标）
        bar_chart_data = {}
        for metric in metrics:
            metric_data = []
            for genre in genres:
                if metric in genre_metrics[genre]:
                    metric_data.append({
                        "genre": genre,
                        "value": genre_metrics[genre][metric]
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
                "total_genres": "int",
                "genres_analyzed": "list",
                "total_texts": "int",
                "texts_by_genre": "dict"
            },
            "genre_metrics": "dict",
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
                "genres": "list",
                "metrics": "list",
                "values": "dict",
                "normalized_values": "dict"
            },
            "genre_rankings": {
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


def test_genre_comparator():
    """测试体裁对比分析器"""
    print("测试体裁词汇对比分析器...")
    
    # 创建测试数据
    test_texts = [
        "This science news article reports on recent research findings in a clear and accessible way.",
        "The academic paper abstract presents a detailed analysis of experimental results using statistical methods.",
        "Another science news piece explains complex concepts for general readers without technical jargon.",
        "A second academic abstract discusses theoretical frameworks and methodological approaches in depth."
    ]
    
    # 创建元数据（包含体裁信息）
    test_metadata = {
        "genre": ["science_news", "academic_paper", "science_news", "academic_paper"],
        "extraction_info": [
            {"genre": "science_news", "source_field": "text"},
            {"genre": "academic_paper", "source_field": "source_pdf.abstract"},
            {"genre": "science_news", "source_field": "text"},
            {"genre": "academic_paper", "source_field": "source_pdf.abstract"}
        ]
    }
    
    # 创建分析器
    comparator = GenreComparator("test_comparator", {
        "calculate_differences": True,
        "calculate_statistical_significance": False,  # 简化测试
        "compare_pairs": True,
        "compare_to_overall": True
    })
    
    # 执行分析
    results = comparator.analyze(test_texts, test_metadata)
    
    if results and "error" not in results:
        print("体裁对比分析器测试成功!")
        
        # 显示基本信息
        stats = results["basic_statistics"]
        print(f"分析的体裁数: {stats['total_genres']}")
        print(f"体裁列表: {', '.join(stats['genres_analyzed'])}")
        
        # 显示体裁指标
        print("\n体裁词汇指标:")
        for genre, metrics in results["genre_metrics"].items():
            print(f"  {genre}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")
                    
        # 显示差异分析
        if "differences_analysis" in results:
            diffs = results["differences_analysis"]
            if "max_min_differences" in diffs:
                print("\n最大差异指标:")
                for metric, info in diffs["max_min_differences"].items():
                    print(f"  {metric}: {info['max_genre']}({info['max_value']:.4f}) vs "
                          f"{info['min_genre']}({info['min_value']:.4f})")
                          
        return True
        
    print(f"体裁对比分析器测试失败: {results.get('error', '未知错误')}")
    return False


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    if test_genre_comparator():
        print("\n体裁词汇对比分析器测试通过!")
    else:
        print("\n体裁词汇对比分析器测试失败!")
