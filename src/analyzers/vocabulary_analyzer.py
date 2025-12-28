#!/usr/bin/env python3
"""
vocabulary_analyzer.py - 词汇分析器

计算词表大小、TTR（Type-Token Ratio）、词频分布等指标。
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import math
import logging

from analyzer_base import BaseAnalyzer, ConfigurableAnalyzer


class VocabularyAnalyzer(ConfigurableAnalyzer):
    """词汇分析器"""
    
    def __init__(self, name: str = "vocabulary_analyzer", config: Optional[Dict[str, Any]] = None):
        """
        初始化词汇分析器
        
        Args:
            name: 分析器名称
            config: 配置字典
        """
        default_config = {
            "language": "en",  # 语言：en, zh, 等
            "min_word_length": 1,  # 最小词长
            "remove_stopwords": False,  # 是否移除停用词
            "stopwords": set(),  # 停用词列表
            "normalize_case": True,  # 是否规范化大小写
            "remove_punctuation": True,  # 是否移除标点
            "calculate_ttr": True,  # 是否计算TTR
            "calculate_mtld": True,  # 是否计算MTLD
            "calculate_word_freq": True,  # 是否计算词频分布
            "top_n_words": 50,  # 显示前N个高频词
            "segment_chinese": True,  # 是否对中文分词
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name, default_config)
        
        # 初始化停用词
        if self.config["remove_stopwords"] and not self.config["stopwords"]:
            self.config["stopwords"] = self._get_default_stopwords()
            
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
    
    def _get_default_stopwords(self) -> set:
        """获取默认停用词列表"""
        # 英文停用词
        english_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
            'should', 'may', 'might', 'must', 'can', 'could', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
            'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers',
            'ours', 'theirs', 'this', 'that', 'these', 'those', 'am', 'are',
            'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may',
            'might', 'must', 'can', 'could'
        }
        
        # 中文停用词
        chinese_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
            '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它',
            '我们', '你们', '他们', '她们', '它们', '这', '那', '这些', '那些',
            '这个', '那个', '这里', '那里', '这样', '那样', '这么', '那么'
        }
        
        if self.config["language"] == "zh":
            return chinese_stopwords
        else:
            return english_stopwords
            
    def _tokenize_text(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            
        Returns:
            词元列表
        """
        if self.config["language"] == "zh" and self.config["segment_chinese"]:
            # 简单的中文分词：按字符分割，但保留常见词语
            return self._simple_chinese_segmentation(text)
        else:
            # 英文或其他语言：按空格和标点分割
            return self._standard_tokenization(text)
            
    def _simple_chinese_segmentation(self, text: str) -> List[str]:
        """简单的中文分词"""
        # 移除标点
        if self.config["remove_punctuation"]:
            text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
            
        # 按空白分割
        tokens = text.split()
        
        # 进一步分割长字符串（如果有）
        result = []
        for token in tokens:
            if len(token) > 1:
                # 尝试按常见模式分割
                result.extend(list(token))
            else:
                result.append(token)
                
        return result
        
    def _standard_tokenization(self, text: str) -> List[str]:
        """标准分词（英文）"""
        # 规范化大小写
        if self.config["normalize_case"]:
            text = text.lower()
            
        # 移除标点
        if self.config["remove_punctuation"]:
            text = re.sub(r'[^\w\s]', ' ', text)
            
        # 按空白分割
        tokens = text.split()
        
        # 过滤短词
        if self.config["min_word_length"] > 1:
            tokens = [t for t in tokens if len(t) >= self.config["min_word_length"]]
            
        return tokens
        
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """移除停用词"""
        if not self.config["remove_stopwords"]:
            return tokens
            
        stopwords = self.config["stopwords"]
        return [t for t in tokens if t not in stopwords]
        
    def analyze(self, texts: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行词汇分析
        
        Args:
            texts: 文本列表
            metadata: 元数据
            
        Returns:
            分析结果字典
        """
        if not self.validate_input(texts):
            return {}
            
        # 预处理文本
        processed_texts = self.preprocess_texts(texts)
        
        # 分词
        all_tokens = []
        tokenized_texts = []
        
        for text in processed_texts:
            tokens = self._tokenize_text(text)
            tokens = self._remove_stopwords(tokens)
            all_tokens.extend(tokens)
            tokenized_texts.append(tokens)
            
        if not all_tokens:
            return {"error": "没有有效的词元"}
            
        # 计算基本统计
        total_tokens = len(all_tokens)
        unique_tokens = set(all_tokens)
        total_types = len(unique_tokens)
        
        # 计算TTR（Type-Token Ratio）
        ttr_results = {}
        if self.config["calculate_ttr"]:
            ttr_results = self._calculate_ttr(all_tokens)
            
        # 计算MTLD（Measure of Textual Lexical Diversity）
        mtld_results = {}
        if self.config["calculate_mtld"]:
            mtld_results = self._calculate_mtld(all_tokens)
            
        # 计算词频分布
        word_freq_results = {}
        if self.config["calculate_word_freq"]:
            word_freq_results = self._calculate_word_frequency(all_tokens)
            
        # 按文本计算统计
        per_text_stats = []
        for i, tokens in enumerate(tokenized_texts):
            if tokens:
                text_stats = {
                    "text_index": i,
                    "token_count": len(tokens),
                    "type_count": len(set(tokens)),
                    "ttr": len(set(tokens)) / len(tokens) if tokens else 0
                }
                per_text_stats.append(text_stats)
                
        return {
            "basic_statistics": {
                "total_tokens": total_tokens,
                "total_types": total_types,
                "type_token_ratio": total_types / total_tokens if total_tokens > 0 else 0,
                "average_tokens_per_text": total_tokens / len(processed_texts) if processed_texts else 0,
                "average_types_per_text": sum(len(set(t)) for t in tokenized_texts) / len(tokenized_texts) if tokenized_texts else 0
            },
            "ttr_analysis": ttr_results,
            "mtld_analysis": mtld_results,
            "word_frequency": word_freq_results,
            "per_text_statistics": per_text_stats,
            "config": self.config
        }
        
    def _calculate_ttr(self, tokens: List[str]) -> Dict[str, Any]:
        """
        计算TTR（Type-Token Ratio）
        
        Args:
            tokens: 词元列表
            
        Returns:
            TTR分析结果
        """
        # 基本TTR
        total_tokens = len(tokens)
        total_types = len(set(tokens))
        basic_ttr = total_types / total_tokens if total_tokens > 0 else 0
        
        # 分段TTR（每100个词）
        segment_size = 100
        segment_ttrs = []
        
        for i in range(0, len(tokens), segment_size):
            segment = tokens[i:i + segment_size]
            if segment:
                segment_types = len(set(segment))
                segment_ttr = segment_types / len(segment)
                segment_ttrs.append({
                    "segment": i // segment_size + 1,
                    "start": i,
                    "end": min(i + segment_size, len(tokens)),
                    "tokens": len(segment),
                    "types": segment_types,
                    "ttr": segment_ttr
                })
                
        # 移动窗口TTR
        window_size = 50
        step_size = 10
        window_ttrs = []
        
        if len(tokens) >= window_size:
            for i in range(0, len(tokens) - window_size + 1, step_size):
                window = tokens[i:i + window_size]
                window_types = len(set(window))
                window_ttr = window_types / window_size
                window_ttrs.append({
                    "window_start": i,
                    "window_end": i + window_size,
                    "ttr": window_ttr
                })
                
        return {
            "basic_ttr": basic_ttr,
            "segment_ttrs": segment_ttrs,
            "window_ttrs": window_ttrs,
            "segment_size": segment_size,
            "window_size": window_size,
            "step_size": step_size
        }
        
    def _calculate_mtld(self, tokens: List[str], threshold: float = 0.72) -> Dict[str, Any]:
        """
        计算MTLD（Measure of Textual Lexical Diversity）
        
        Args:
            tokens: 词元列表
            threshold: MTLD阈值（通常为0.72）
            
        Returns:
            MTLD分析结果
        """
        if not tokens:
            return {"mtld": 0, "factors": 0}
            
        # 正向MTLD
        forward_mtld = self._calculate_mtld_direction(tokens, threshold)
        
        # 反向MTLD
        reverse_tokens = list(reversed(tokens))
        reverse_mtld = self._calculate_mtld_direction(reverse_tokens, threshold)
        
        # 平均MTLD
        avg_mtld = (forward_mtld + reverse_mtld) / 2 if forward_mtld > 0 and reverse_mtld > 0 else max(forward_mtld, reverse_mtld)
        
        return {
            "mtld_forward": forward_mtld,
            "mtld_reverse": reverse_mtld,
            "mtld_average": avg_mtld,
            "threshold": threshold
        }
        
    def _calculate_mtld_direction(self, tokens: List[str], threshold: float) -> float:
        """
        计算单向MTLD
        
        Args:
            tokens: 词元列表
            threshold: MTLD阈值
            
        Returns:
            MTLD值
        """
        if not tokens:
            return 0.0
            
        factors = 0
        start = 0
        
        while start < len(tokens):
            seen_types = set()
            ttr = 1.0
            length = 0
            
            # 寻找因子
            for i in range(start, len(tokens)):
                seen_types.add(tokens[i])
                length += 1
                ttr = len(seen_types) / length
                
                if ttr <= threshold:
                    factors += 1
                    start = i + 1
                    break
                    
            # 如果到达文本末尾且ttr > threshold
            if ttr > threshold:
                # 计算部分因子
                partial_factor = (1 - ttr) / (1 - threshold) if threshold < 1.0 else 0
                factors += partial_factor
                break
                
        # 计算MTLD
        mtld = len(tokens) / factors if factors > 0 else 0
        return mtld
        
    def _calculate_word_frequency(self, tokens: List[str]) -> Dict[str, Any]:
        """
        计算词频分布
        
        Args:
            tokens: 词元列表
            
        Returns:
            词频分析结果
        """
        # 计算词频
        word_freq = Counter(tokens)
        total_tokens = len(tokens)
        
        # 高频词
        top_n = self.config["top_n_words"]
        top_words = word_freq.most_common(top_n)
        
        # 词频分布统计
        freq_distribution = Counter(word_freq.values())
        
        # 计算Zipf分布参数
        ranks = list(range(1, len(word_freq) + 1))
        frequencies = [freq for _, freq in word_freq.most_common()]
        
        # 尝试拟合幂律分布
        zipf_params = self._fit_zipf_law(ranks, frequencies)
        
        return {
            "total_unique_words": len(word_freq),
            "top_words": [{"word": word, "frequency": freq, "percentage": freq/total_tokens*100} 
                         for word, freq in top_words],
            "frequency_distribution": [{"frequency": freq, "count": count} 
                                      for freq, count in freq_distribution.most_common()],
            "zipf_analysis": zipf_params,
            "hapax_legomena": sum(1 for freq in word_freq.values() if freq == 1),  # 出现一次的词
            "dis_legomena": sum(1 for freq in word_freq.values() if freq == 2),  # 出现两次的词
            "vocabulary_richness": len(word_freq) / math.sqrt(total_tokens) if total_tokens > 0 else 0
        }
        
    def _fit_zipf_law(self, ranks: List[int], frequencies: List[int]) -> Dict[str, Any]:
        """
        拟合Zipf定律
        
        Args:
            ranks: 排名列表
            frequencies: 频率列表
            
        Returns:
            Zipf定律参数
        """
        if len(ranks) < 2 or len(frequencies) < 2:
            return {"error": "数据不足"}
            
        # 取对数
        log_ranks = [math.log(r) for r in ranks]
        log_freqs = [math.log(f) for f in frequencies]
        
        # 简单线性回归
        n = len(ranks)
        sum_x = sum(log_ranks)
        sum_y = sum(log_freqs)
        sum_xy = sum(x * y for x, y in zip(log_ranks, log_freqs))
        sum_x2 = sum(x * x for x in log_ranks)
        
        # 计算斜率和截距
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        intercept = (sum_y - slope * sum_x) / n
        
        # 计算R²
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in log_freqs)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_ranks, log_freqs))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "zipf_constant": -slope,  # Zipf定律中的指数
            "predicted_frequency_for_rank_1": math.exp(intercept)
        }
        
    def get_required_fields(self) -> List[str]:
        """获取需要的字段"""
        return ["text"]  # 只需要文本字段
        
    def get_result_schema(self) -> Dict[str, Any]:
        """获取结果模式"""
        return {
            "basic_statistics": {
                "total_tokens": "int",
                "total_types": "int", 
                "type_token_ratio": "float",
                "average_tokens_per_text": "float",
                "average_types_per_text": "float"
            },
            "ttr_analysis": {
                "basic_ttr": "float",
                "segment_ttrs": "list",
                "window_ttrs": "list"
            },
            "mtld_analysis": {
                "mtld_forward": "float",
                "mtld_reverse": "float", 
                "mtld_average": "float"
            },
            "word_frequency": {
                "total_unique_words": "int",
                "top_words": "list",
                "frequency_distribution": "list",
                "zipf_analysis": "dict",
                "hapax_legomena": "int",
                "dis_legomena": "int",
                "vocabulary_richness": "float"
            },
            "per_text_statistics": "list",
            "config": "dict"
        }
