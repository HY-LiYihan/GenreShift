#!/usr/bin/env python3
"""
academic_vocabulary_analyzer.py - 学术词汇分析器

分析学术词汇比例、学科特定词汇等。
"""

import re
from typing import Dict, List, Any, Optional, Set
from collections import Counter, defaultdict
import json
import math
import logging
from pathlib import Path

from analyzer_base import BaseAnalyzer, ConfigurableAnalyzer


class AcademicVocabularyAnalyzer(ConfigurableAnalyzer):
    """学术词汇分析器"""
    
    def __init__(self, name: str = "academic_vocabulary_analyzer", config: Optional[Dict[str, Any]] = None):
        """
        初始化学术词汇分析器
        
        Args:
            name: 分析器名称
            config: 配置字典
        """
        default_config = {
            "language": "en",  # 语言
            "academic_word_lists": {},  # 学术词表
            "field_specific_vocabularies": {},  # 学科特定词汇
            "calculate_academic_ratio": True,  # 是否计算学术词汇比例
            "calculate_field_specific_ratio": True,  # 是否计算学科特定词汇比例
            "normalize_case": True,  # 是否规范化大小写
            "remove_stopwords": False,  # 是否移除停用词
            "min_word_length": 3,  # 最小词长
            "load_default_lists": True,  # 是否加载默认词表
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name, default_config)
        
        # 加载默认词表
        if self.config["load_default_lists"]:
            self._load_default_word_lists()
            
    def _load_default_word_lists(self):
        """加载默认词表"""
        # 尝试从文件加载，如果文件不存在则使用内置词表
        try:
            # 这里可以添加从文件加载词表的逻辑
            pass
        except:
            # 使用内置默认词表
            self._load_builtin_word_lists()
            
    def _load_builtin_word_lists(self):
        """加载内置词表"""
        # 通用学术词汇（AWL - Academic Word List）
        academic_words = {
            "analyze", "approach", "area", "assess", "assume", "authority", "available",
            "benefit", "concept", "consist", "constitute", "context", "contract", "create",
            "data", "define", "derive", "distribute", "economy", "environment", "establish",
            "estimate", "evident", "export", "factor", "finance", "formula", "function",
            "identify", "income", "indicate", "individual", "interpret", "involve", "issue",
            "labor", "legal", "legislate", "major", "method", "occur", "percent", "period",
            "policy", "principle", "procedure", "process", "require", "research", "respond",
            "role", "section", "sector", "significant", "similar", "source", "specific",
            "structure", "theory", "vary", "achieve", "acquire", "administrate", "affect",
            "appropriate", "aspect", "assist", "category", "chapter", "commission", "community",
            "complex", "compute", "conclude", "conduct", "consequent", "construct", "consume",
            "credit", "culture", "design", "distinct", "element", "equate", "evaluate",
            "feature", "final", "focus", "impact", "injure", "institute", "invest", "item",
            "journal", "maintain", "normal", "obtain", "participate", "perceive", "positive",
            "potential", "previous", "primary", "purchase", "range", "region", "regulate",
            "relevant", "reside", "resource", "restrict", "secure", "seek", "select",
            "site", "strategy", "survey", "text", "tradition", "transfer"
        }
        
        # 学科特定词汇示例
        field_vocabularies = {
            "science": {
                "method", "experiment", "hypothesis", "theory", "data", "analysis",
                "result", "conclusion", "variable", "control", "observation", "measurement",
                "evidence", "proof", "law", "principle", "model", "simulation", "prediction"
            },
            "humanities": {
                "interpretation", "analysis", "context", "text", "discourse", "narrative",
                "representation", "identity", "culture", "society", "history", "philosophy",
                "critique", "theory", "methodology", "perspective", "framework", "discussion"
            },
            "social_sciences": {
                "survey", "interview", "data", "analysis", "theory", "model", "variable",
                "correlation", "causation", "population", "sample", "statistic", "hypothesis",
                "methodology", "framework", "concept", "construct", "measure", "scale"
            }
        }
        
        self.config["academic_word_lists"] = {
            "general": academic_words,
            "awl": academic_words  # 别名
        }
        
        self.config["field_specific_vocabularies"] = field_vocabularies
        
    def _tokenize_text(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            
        Returns:
            词元列表
        """
        # 规范化大小写
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
        
    def _identify_academic_words(self, tokens: List[str]) -> Dict[str, List[str]]:
        """
        识别学术词汇
        
        Args:
            tokens: 词元列表
            
        Returns:
            识别的学术词汇分类
        """
        academic_words = self.config["academic_word_lists"]
        field_vocabularies = self.config["field_specific_vocabularies"]
        
        identified = {
            "general_academic": [],
            "field_specific": defaultdict(list)
        }
        
        token_set = set(tokens)
        
        # 识别通用学术词汇
        general_lists = academic_words.get("general", set()) | academic_words.get("awl", set())
        for word in token_set:
            if word in general_lists:
                identified["general_academic"].append(word)
                
        # 识别学科特定词汇
        for field, vocabulary in field_vocabularies.items():
            field_words = token_set & vocabulary
            if field_words:
                identified["field_specific"][field] = list(field_words)
                
        return identified
        
    def analyze(self, texts: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行学术词汇分析
        
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
            all_tokens.extend(tokens)
            tokenized_texts.append(tokens)
            
        if not all_tokens:
            return {"error": "没有有效的词元"}
            
        # 识别学术词汇
        academic_words = self._identify_academic_words(all_tokens)
        
        # 计算基本统计
        total_tokens = len(all_tokens)
        unique_tokens = set(all_tokens)
        total_types = len(unique_tokens)
        
        # 计算学术词汇比例
        academic_ratios = {}
        if self.config["calculate_academic_ratio"]:
            academic_ratios = self._calculate_academic_ratios(all_tokens, academic_words)
            
        # 计算学科特定词汇比例
        field_specific_ratios = {}
        if self.config["calculate_field_specific_ratio"]:
            field_specific_ratios = self._calculate_field_specific_ratios(all_tokens, academic_words)
            
        # 按文本计算统计
        per_text_stats = []
        for i, tokens in enumerate(tokenized_texts):
            if tokens:
                text_academic_words = self._identify_academic_words(tokens)
                text_academic_ratio = self._calculate_academic_ratios(tokens, text_academic_words)
                
                text_stats = {
                    "text_index": i,
                    "token_count": len(tokens),
                    "academic_word_count": text_academic_ratio.get("academic_token_count", 0),
                    "academic_word_ratio": text_academic_ratio.get("academic_token_ratio", 0),
                    "general_academic_count": text_academic_ratio.get("general_academic_token_count", 0),
                    "general_academic_ratio": text_academic_ratio.get("general_academic_token_ratio", 0)
                }
                per_text_stats.append(text_stats)
                
        # 学术词汇分布
        academic_word_freq = Counter()
        for word in academic_words["general_academic"]:
            academic_word_freq[word] += all_tokens.count(word)
            
        # 学科词汇分布
        field_word_freq = {}
        for field, words in academic_words["field_specific"].items():
            field_counter = Counter()
            for word in words:
                field_counter[word] += all_tokens.count(word)
            field_word_freq[field] = dict(field_counter.most_common(20))
            
        return {
            "basic_statistics": {
                "total_tokens": total_tokens,
                "total_types": total_types,
                "academic_word_types": len(academic_words["general_academic"]),
                "field_specific_word_types": sum(len(words) for words in academic_words["field_specific"].values())
            },
            "academic_word_ratios": academic_ratios,
            "field_specific_ratios": field_specific_ratios,
            "academic_word_frequency": {
                "top_academic_words": [{"word": word, "frequency": freq} 
                                      for word, freq in academic_word_freq.most_common(50)],
                "field_specific_words": field_word_freq
            },
            "identified_words": {
                "general_academic": sorted(academic_words["general_academic"]),
                "field_specific": {field: sorted(words) for field, words in academic_words["field_specific"].items()}
            },
            "per_text_statistics": per_text_stats,
            "config": {
                "language": self.config["language"],
                "calculate_academic_ratio": self.config["calculate_academic_ratio"],
                "calculate_field_specific_ratio": self.config["calculate_field_specific_ratio"]
            }
        }
        
    def _calculate_academic_ratios(self, tokens: List[str], academic_words: Dict) -> Dict[str, Any]:
        """
        计算学术词汇比例
        
        Args:
            tokens: 词元列表
            academic_words: 识别的学术词汇
            
        Returns:
            学术词汇比例统计
        """
        total_tokens = len(tokens)
        
        # 通用学术词汇
        general_academic_words = academic_words["general_academic"]
        general_academic_token_count = sum(tokens.count(word) for word in general_academic_words)
        
        # 所有学术词汇（包括学科特定）
        all_academic_words = set(general_academic_words)
        for field_words in academic_words["field_specific"].values():
            all_academic_words.update(field_words)
            
        academic_token_count = sum(tokens.count(word) for word in all_academic_words)
        
        # 计算比例
        general_academic_ratio = general_academic_token_count / total_tokens if total_tokens > 0 else 0
        academic_ratio = academic_token_count / total_tokens if total_tokens > 0 else 0
        
        return {
            "academic_token_count": academic_token_count,
            "academic_token_ratio": academic_ratio,
            "academic_type_count": len(all_academic_words),
            "academic_type_ratio": len(all_academic_words) / len(set(tokens)) if tokens else 0,
            "general_academic_token_count": general_academic_token_count,
            "general_academic_token_ratio": general_academic_ratio,
            "general_academic_type_count": len(general_academic_words),
            "general_academic_type_ratio": len(general_academic_words) / len(set(tokens)) if tokens else 0
        }
        
    def _calculate_field_specific_ratios(self, tokens: List[str], academic_words: Dict) -> Dict[str, Any]:
        """
        计算学科特定词汇比例
        
        Args:
            tokens: 词元列表
            academic_words: 识别的学术词汇
            
        Returns:
            学科特定词汇比例统计
        """
        total_tokens = len(tokens)
        field_ratios = {}
        
        for field, words in academic_words["field_specific"].items():
            if words:
                field_token_count = sum(tokens.count(word) for word in words)
                field_ratio = field_token_count / total_tokens if total_tokens > 0 else 0
                
                field_ratios[field] = {
                    "token_count": field_token_count,
                    "token_ratio": field_ratio,
                    "type_count": len(words),
                    "type_ratio": len(words) / len(set(tokens)) if tokens else 0,
                    "words": sorted(words)
                }
                
        # 总体学科词汇统计
        all_field_words = set()
        for words in academic_words["field_specific"].values():
            all_field_words.update(words)
            
        total_field_token_count = sum(tokens.count(word) for word in all_field_words)
        total_field_ratio = total_field_token_count / total_tokens if total_tokens > 0 else 0
        
        return {
            "field_specific": field_ratios,
            "overall": {
                "total_field_token_count": total_field_token_count,
                "total_field_token_ratio": total_field_ratio,
                "total_field_type_count": len(all_field_words),
                "total_field_type_ratio": len(all_field_words) / len(set(tokens)) if tokens else 0
            }
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
                "academic_word_types": "int",
                "field_specific_word_types": "int"
            },
            "academic_word_ratios": {
                "academic_token_count": "int",
                "academic_token_ratio": "float",
                "academic_type_count": "int",
                "academic_type_ratio": "float",
                "general_academic_token_count": "int",
                "general_academic_token_ratio": "float",
                "general_academic_type_count": "int",
                "general_academic_type_ratio": "float"
            },
            "field_specific_ratios": {
                "field_specific": "dict",
                "overall": "dict"
            },
            "academic_word_frequency": {
                "top_academic_words": "list",
                "field_specific_words": "dict"
            },
            "identified_words": {
                "general_academic": "list",
                "field_specific": "dict"
            },
            "per_text_statistics": "list",
            "config": "dict"
        }
