#!/usr/bin/env python3
"""
enhanced_academic_analyzer.py - 增强版学术词汇分析器

支持从下载的词表加载学术词汇和学科词汇，提供更准确的学科分析。
"""

import re
import json
from typing import Dict, List, Any, Optional, Set
from collections import Counter, defaultdict
from pathlib import Path
import logging

from analyzer_base import ConfigurableAnalyzer

# 尝试导入词表下载器
try:
    from wordlists.wordlist_downloader import WordlistDownloader
    WORDLIST_DOWNLOADER_AVAILABLE = True
except ImportError:
    WORDLIST_DOWNLOADER_AVAILABLE = False
    print("警告: wordlist_downloader 不可用，将使用内置词表")


class EnhancedAcademicAnalyzer(ConfigurableAnalyzer):
    """增强版学术词汇分析器，支持外部词表"""
    
    def __init__(self, name: str = "enhanced_academic_analyzer", config: Optional[Dict[str, Any]] = None):
        """
        初始化增强版学术词汇分析器
        
        Args:
            name: 分析器名称
            config: 配置字典
        """
        default_config = {
            "language": "en",
            "wordlist_cache_dir": "data/wordlists",
            "use_downloaded_wordlists": True,
            "download_if_missing": True,
            "academic_wordlist_ids": ["awl"],
            "discipline_wordlist_ids": [],  # 学科词表ID列表
            "calculate_academic_ratio": True,
            "calculate_discipline_ratios": True,
            "calculate_cross_discipline": True,
            "normalize_case": True,
            "min_word_length": 3,
            "use_nltk_wordlists": True,
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name, default_config)
        
        # 词表存储
        self.academic_wordlists: Dict[str, Set[str]] = {}
        self.discipline_wordlists: Dict[str, Set[str]] = {}
        self.nltk_wordlists: Dict[str, Set[str]] = {}
        
        # 加载词表
        self._load_wordlists()
        
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
        
    def _load_wordlists(self):
        """加载所有词表"""
        # 加载NLTK词表
        if self.config["use_nltk_wordlists"]:
            self._load_nltk_wordlists()
            
        # 加载下载的词表
        if self.config["use_downloaded_wordlists"] and WORDLIST_DOWNLOADER_AVAILABLE:
            self._load_downloaded_wordlists()
        else:
            # 使用内置词表作为后备
            self._load_builtin_wordlists()
            
    def _load_nltk_wordlists(self):
        """加载NLTK词表"""
        try:
            import nltk
            from nltk.corpus import stopwords, words, wordnet
            
            # 停用词表
            nltk_stopwords = set(stopwords.words('english'))
            self.nltk_wordlists['stopwords'] = nltk_stopwords
            
            # 英语词汇表
            nltk_words = set(words.words())
            self.nltk_wordlists['words'] = nltk_words
            
            # WordNet词汇
            wordnet_words = set()
            for synset in wordnet.all_synsets():
                for lemma in synset.lemmas():
                    wordnet_words.add(lemma.name().replace('_', ' '))
            self.nltk_wordlists['wordnet'] = wordnet_words
            
            self.logger.info(f"加载NLTK词表: 停用词 {len(nltk_stopwords)}, 词汇 {len(nltk_words)}, WordNet {len(wordnet_words)}")
            
        except Exception as e:
            self.logger.warning(f"加载NLTK词表失败: {e}")
            
    def _load_downloaded_wordlists(self):
        """加载下载的词表"""
        try:
            downloader = WordlistDownloader(self.config["wordlist_cache_dir"])
            
            # 下载缺失的词表
            if self.config["download_if_missing"]:
                downloader.download_all_wordlists(force_download=False)
                
            # 加载学术词表
            for wordlist_id in self.config["academic_wordlist_ids"]:
                cache_files = list(Path(self.config["wordlist_cache_dir"]).glob(f"{wordlist_id}_*.csv"))
                cache_files.extend(list(Path(self.config["wordlist_cache_dir"]).glob(f"{wordlist_id}_*.txt")))
                
                if cache_files:
                    latest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
                    words = downloader.parse_wordlist_file(latest_file)
                    self.academic_wordlists[wordlist_id] = set(words)
                    self.logger.info(f"加载学术词表 {wordlist_id}: {len(words)} 词")
                else:
                    self.logger.warning(f"学术词表 {wordlist_id} 未找到")
                    
            # 加载学科词表
            for wordlist_id in self.config["discipline_wordlist_ids"]:
                cache_files = list(Path(self.config["wordlist_cache_dir"]).glob(f"{wordlist_id}_*.txt"))
                cache_files.extend(list(Path(self.config["wordlist_cache_dir"]).glob(f"discipline_{wordlist_id}_*.txt")))
                
                if cache_files:
                    latest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
                    words = downloader.parse_wordlist_file(latest_file)
                    self.discipline_wordlists[wordlist_id] = set(words)
                    self.logger.info(f"加载学科词表 {wordlist_id}: {len(words)} 词")
                else:
                    self.logger.warning(f"学科词表 {wordlist_id} 未找到")
                    
        except Exception as e:
            self.logger.error(f"加载下载的词表失败: {e}")
            self._load_builtin_wordlists()  # 回退到内置词表
            
    def _load_builtin_wordlists(self):
        """加载内置词表（后备）"""
        # 内置学术词汇（AWL子集）
        builtin_academic = {
            "analyze", "approach", "area", "assess", "assume", "authority", "available",
            "benefit", "concept", "consist", "constitute", "context", "contract", "create",
            "data", "define", "derive", "distribute", "economy", "environment", "establish",
            "estimate", "evident", "export", "factor", "finance", "formula", "function",
            "identify", "income", "indicate", "individual", "interpret", "involve", "issue",
            "labor", "legal", "legislate", "major", "method", "occur", "percent", "period",
            "policy", "principle", "procedure", "process", "require", "research", "respond",
            "role", "section", "sector", "significant", "similar", "source", "specific",
            "structure", "theory", "vary"
        }
        
        # 内置学科词汇
        builtin_disciplines = {
            "science": {
                "method", "experiment", "hypothesis", "theory", "data", "analysis",
                "result", "conclusion", "variable", "control", "observation", "measurement"
            },
            "humanities": {
                "interpretation", "analysis", "context", "text", "discourse", "narrative",
                "representation", "identity", "culture", "society", "history", "philosophy"
            },
            "social_sciences": {
                "survey", "interview", "data", "analysis", "theory", "model", "variable",
                "correlation", "causation", "population", "sample", "statistic", "hypothesis"
            }
        }
        
        self.academic_wordlists["builtin"] = builtin_academic
        self.discipline_wordlists = builtin_disciplines
        
        self.logger.info(f"加载内置词表: 学术词汇 {len(builtin_academic)}, 学科 {len(builtin_disciplines)}")
        
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
        
    def _identify_words(self, tokens: List[str]) -> Dict[str, Any]:
        """
        识别各类词汇
        
        Args:
            tokens: 词元列表
            
        Returns:
            识别的词汇分类
        """
        token_set = set(tokens)
        
        identified = {
            "academic": defaultdict(list),
            "discipline": defaultdict(list),
            "common": [],
            "unique": list(token_set)
        }
        
        # 识别学术词汇
        for wordlist_id, wordlist in self.academic_wordlists.items():
            academic_words = token_set & wordlist
            if academic_words:
                identified["academic"][wordlist_id] = list(academic_words)
                
        # 识别学科词汇
        for discipline, wordlist in self.discipline_wordlists.items():
            discipline_words = token_set & wordlist
            if discipline_words:
                identified["discipline"][discipline] = list(discipline_words)
                
        # 识别常用词（NLTK词汇）
        if 'words' in self.nltk_wordlists:
            common_words = token_set & self.nltk_wordlists['words']
            identified["common"] = list(common_words)
            
        return identified
        
    def analyze(self, texts: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行增强版学术词汇分析
        
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
            
        # 识别词汇
        identified_words = self._identify_words(all_tokens)
        
        # 计算基本统计
        total_tokens = len(all_tokens)
        unique_tokens = set(all_tokens)
        total_types = len(unique_tokens)
        
        # 计算学术词汇比例
        academic_ratios = {}
        if self.config["calculate_academic_ratio"]:
            academic_ratios = self._calculate_academic_ratios(all_tokens, identified_words)
            
        # 计算学科比例
        discipline_ratios = {}
        if self.config["calculate_discipline_ratios"]:
            discipline_ratios = self._calculate_discipline_ratios(all_tokens, identified_words)
            
        # 计算交叉学科分析
        cross_discipline = {}
        if self.config["calculate_cross_discipline"]:
            cross_discipline = self._calculate_cross_discipline(all_tokens, identified_words)
            
        # 按文本计算统计
        per_text_stats = []
        for i, tokens in enumerate(tokenized_texts):
            if tokens:
                text_identified = self._identify_words(tokens)
                text_academic = self._calculate_academic_ratios(tokens, text_identified)
                text_discipline = self._calculate_discipline_ratios(tokens, text_identified)
                
                text_stats = {
                    "text_index": i,
                    "token_count": len(tokens),
                    "academic_ratio": text_academic.get("overall_ratio", 0),
                    "discipline_distribution": text_discipline.get("distribution", {}),
                    "primary_discipline": text_discipline.get("primary_discipline", "unknown")
                }
                per_text_stats.append(text_stats)
                
        # 词汇频率分析
        word_freq = Counter(all_tokens)
        
        # 学术词汇频率
        academic_word_freq = {}
        for wordlist_id, words in identified_words["academic"].items():
            freq_dict = {}
            for word in words:
                freq_dict[word] = word_freq[word]
            academic_word_freq[wordlist_id] = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:20])
            
        # 学科词汇频率
        discipline_word_freq = {}
        for discipline, words in identified_words["discipline"].items():
            freq_dict = {}
            for word in words:
                freq_dict[word] = word_freq[word]
            discipline_word_freq[discipline] = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:20])
            
        return {
            "basic_statistics": {
                "total_tokens": total_tokens,
                "total_types": total_types,
                "academic_types": sum(len(words) for words in identified_words["academic"].values()),
                "discipline_types": sum(len(words) for words in identified_words["discipline"].values()),
                "common_types": len(identified_words["common"]),
                "unique_types": len(identified_words["unique"])
            },
            "academic_analysis": academic_ratios,
            "discipline_analysis": discipline_ratios,
            "cross_discipline_analysis": cross_discipline,
            "word_frequency": {
                "academic_words": academic_word_freq,
                "discipline_words": discipline_word_freq,
                "top_common_words": [{"word": word, "frequency": freq} 
                                   for word, freq in word_freq.most_common(50)]
            },
            "identified_words": {
                "academic": {k: sorted(v) for k, v in identified_words["academic"].items()},
                "discipline": {k: sorted(v) for k, v in identified_words["discipline"].items()},
                "common": sorted(identified_words["common"])[:100]
            },
            "per_text_statistics": per_text_stats,
            "config": {
                "academic_wordlists_used": list(self.academic_wordlists.keys()),
                "discipline_wordlists_used": list(self.discipline_wordlists.keys()),
                "nltk_wordlists_used": list(self.nltk_wordlists.keys())
            }
        }
        
    def _calculate_academic_ratios(self, tokens: List[str], identified_words: Dict) -> Dict[str, Any]:
        """计算学术词汇比例"""
        total_tokens = len(tokens)
        
        # 收集所有学术词汇
        all_academic_words = set()
        for words in identified_words["academic"].values():
            all_academic_words.update(words)
            
        # 计算统计
        academic_token_count = sum(tokens.count(word) for word in all_academic_words)
        academic_ratio = academic_token_count / total_tokens if total_tokens > 0 else 0
        
        # 按词表分类统计
        by_wordlist = {}
        for wordlist_id, words in identified_words["academic"].items():
            if words:
                token_count = sum(tokens.count(word) for word in words)
                by_wordlist[wordlist_id] = {
                    "type_count": len(words),
                    "token_count": token_count,
                    "token_ratio": token_count / total_tokens if total_tokens > 0 else 0
                }
                
        return {
            "overall_ratio": academic_ratio,
            "total_academic_tokens": academic_token_count,
            "total_academic_types": len(all_academic_words),
            "by_wordlist": by_wordlist
        }
        
    def _calculate_discipline_ratios(self, tokens: List[str], identified_words: Dict) -> Dict[str, Any]:
        """计算学科词汇比例"""
        total_tokens = len(tokens)
        
        # 按学科统计
        by_discipline = {}
        all_discipline_words = set()
        
        for discipline, words in identified_words["discipline"].items():
            if words:
                token_count = sum(tokens.count(word) for word in words)
                all_discipline_words.update(words)
                
                by_discipline[discipline] = {
                    "type_count": len(words),
                    "token_count": token_count,
                    "token_ratio": token_count / total_tokens if total_tokens > 0 else 0,
                    "words": sorted(words)
                }
                
        # 总体统计
        total_discipline_tokens = sum(tokens.count(word) for word in all_discipline_words)
        total_discipline_ratio = total_discipline_tokens / total_tokens if total_tokens > 0 else 0
        
        # 主要学科（基于词数）
        primary_discipline = None
        if by_discipline:
            primary_discipline = max(by_discipline.items(), key=lambda x: x[1]["token_count"])[0]
            
        # 学科分布（标准化）
        distribution = {}
        if by_discipline:
            total_discipline_tokens_all = sum(info["token_count"] for info in by_discipline.values())
            if total_discipline_tokens_all > 0:
                for discipline, info in by_discipline.items():
                    distribution[discipline] = info["token_count"] / total_discipline_tokens_all
                    
        return {
            "by_discipline": by_discipline,
            "overall": {
                "total_discipline_tokens": total_discipline_tokens,
                "total_discipline_ratio": total_discipline_ratio,
                "total_discipline_types": len(all_discipline_words)
            },
            "primary_discipline": primary_discipline,
            "distribution": distribution
        }
        
    def _calculate_cross_discipline(self, tokens: List[str], identified_words: Dict) -> Dict[str, Any]:
        """计算交叉学科分析"""
        # 学科共现分析
        discipline_cooccurrence = defaultdict(int)
        
        # 获取所有学科
        disciplines = list(identified_words["discipline"].keys())
        
        # 初始化共现矩阵
        cooccurrence_matrix = {}
        for i, disc1 in enumerate(disciplines):
            for disc2 in disciplines[i:]:
                key = f"{disc1}-{disc2}" if disc1 != disc2 else disc1
                cooccurrence_matrix[key] = 0
                
        # 计算共现（简化版：文本级别的学科共现）
        # 在实际实现中，应该在文本级别计算
        
        # 学科相似度（基于共享词汇）
        discipline_similarity = {}
        for i, disc1 in enumerate(disciplines):
            for disc2 in disciplines[i+1:]:
                words1 = set(identified_words["discipline"].get(disc1, []))
                words2 = set(identified_words["discipline"].get(disc2, []))
                
                if words1 and words2:
                    # Jaccard相似度
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    similarity = intersection / union if union > 0 else 0
                    
                    discipline_similarity[f"{disc1}-{disc2}"] = {
                        "similarity": similarity,
                        "shared_words": sorted(list(words1 & words2))[:10],
                        "shared_count": intersection
                    }
                    
        # 学科独特性（学科特有词汇比例）
        discipline_uniqueness = {}
        for discipline, words in identified_words["discipline"].items():
            if words:
                # 计算该学科词汇在其他学科中的出现情况
                other_disciplines = [d for d in disciplines if d != discipline]
                unique_words = set(words)
                
                for other_disc in other_disciplines:
                    other_words = set(identified_words["discipline"].get(other_disc, []))
                    unique_words -= other_words
                    
                uniqueness_ratio = len(unique_words) / len(words) if words else 0
                
                discipline_uniqueness[discipline] = {
                    "unique_words": sorted(list(unique_words))[:20],
                    "unique_count": len(unique_words),
                    "total_words": len(words),
                    "uniqueness_ratio": uniqueness_ratio
                }
                
        return {
            "cooccurrence_matrix": dict(cooccurrence_matrix),
            "discipline_similarity": discipline_similarity,
            "discipline_uniqueness": discipline_uniqueness,
            "discipline_overlap": self._calculate_discipline_overlap(identified_words)
        }
        
    def _calculate_discipline_overlap(self, identified_words: Dict) -> Dict[str, Any]:
        """计算学科重叠度"""
        disciplines = list(identified_words["discipline"].keys())
        
        if len(disciplines) < 2:
            return {"message": "需要至少2个学科进行重叠分析"}
            
        # 创建词汇-学科映射
        word_to_disciplines = defaultdict(set)
        for discipline, words in identified_words["discipline"].items():
            for word in words:
                word_to_disciplines[word].add(discipline)
                
        # 分析词汇分布
        overlap_stats = {
            "unique_words_by_discipline": {},
            "shared_words": {},
            "overlap_matrix": {}
        }
        
        # 每个学科的独特词汇
        for discipline in disciplines:
            unique_words = []
            for word, disc_set in word_to_disciplines.items():
                if disc_set == {discipline}:
                    unique_words.append(word)
                    
            overlap_stats["unique_words_by_discipline"][discipline] = {
                "count": len(unique_words),
                "words": sorted(unique_words)[:20]
            }
            
        # 共享词汇分析
        for i, disc1 in enumerate(disciplines):
            for disc2 in disciplines[i+1:]:
                shared_words = []
                for word, disc_set in word_to_disciplines.items():
                    if {disc1, disc2}.issubset(disc_set):
                        shared_words.append(word)
                        
                key = f"{disc1}-{disc2}"
                overlap_stats["shared_words"][key] = {
                    "count": len(shared_words),
                    "words": sorted(shared_words)[:20]
                }
                
        # 重叠矩阵
        for disc1 in disciplines:
            overlap_stats["overlap_matrix"][disc1] = {}
            for disc2 in disciplines:
                if disc1 == disc2:
                    overlap_stats["overlap_matrix"][disc1][disc2] = 1.0
                else:
                    words1 = set(identified_words["discipline"].get(disc1, []))
                    words2 = set(identified_words["discipline"].get(disc2, []))
                    
                    if words1 and words2:
                        overlap = len(words1 & words2) / len(words1 | words2)
                    else:
                        overlap = 0.0
                        
                    overlap_stats["overlap_matrix"][disc1][disc2] = overlap
                    
        return overlap_stats
        
    def get_required_fields(self) -> List[str]:
        """获取需要的字段"""
        return ["text"]
        
    def get_result_schema(self) -> Dict[str, Any]:
        """获取结果模式"""
        return {
            "basic_statistics": {
                "total_tokens": "int",
                "total_types": "int",
                "academic_types": "int",
                "discipline_types": "int",
                "common_types": "int",
                "unique_types": "int"
            },
            "academic_analysis": {
                "overall_ratio": "float",
                "total_academic_tokens": "int",
                "total_academic_types": "int",
                "by_wordlist": "dict"
            },
            "discipline_analysis": {
                "by_discipline": "dict",
                "overall": "dict",
                "primary_discipline": "str",
                "distribution": "dict"
            },
            "cross_discipline_analysis": {
                "cooccurrence_matrix": "dict",
                "discipline_similarity": "dict",
                "discipline_uniqueness": "dict",
                "discipline_overlap": "dict"
            },
            "word_frequency": {
                "academic_words": "dict",
                "discipline_words": "dict",
                "top_common_words": "list"
            },
            "identified_words": {
                "academic": "dict",
                "discipline": "dict",
                "common": "list"
            },
            "per_text_statistics": "list",
            "config": "dict"
        }
        
    def get_wordlist_info(self) -> Dict[str, Any]:
        """获取词表信息"""
        return {
            "academic_wordlists": {
                name: len(words) for name, words in self.academic_wordlists.items()
            },
            "discipline_wordlists": {
                name: len(words) for name, words in self.discipline_wordlists.items()
            },
            "nltk_wordlists": {
                name: len(words) for name, words in self.nltk_wordlists.items()
            },
            "config": {
                "academic_wordlist_ids": self.config["academic_wordlist_ids"],
                "discipline_wordlist_ids": self.config["discipline_wordlist_ids"],
                "use_downloaded_wordlists": self.config["use_downloaded_wordlists"],
                "download_if_missing": self.config["download_if_missing"]
            }
        }


def test_enhanced_analyzer():
    """测试增强版分析器"""
    print("测试增强版学术词汇分析器...")
    
    # 创建测试文本
    test_texts = [
        "This research analyzes scientific data using statistical methods and theoretical frameworks.",
        "Humanities scholars interpret historical texts through critical analysis and cultural context.",
        "Social sciences employ surveys and interviews to study population dynamics and social behavior.",
        "The methodology combines experimental design with quantitative analysis and qualitative interpretation."
    ]
    
    # 创建分析器
    analyzer = EnhancedAcademicAnalyzer("test_enhanced", {
        "use_downloaded_wordlists": False,  # 测试时使用内置词表
        "calculate_academic_ratio": True,
        "calculate_discipline_ratios": True,
        "calculate_cross_discipline": True
    })
    
    # 执行分析
    results = analyzer.analyze(test_texts)
    
    if results and "error" not in results:
        print("增强版分析器测试成功!")
        
        # 显示基本信息
        stats = results["basic_statistics"]
        print(f"总词数: {stats['total_tokens']}")
        print(f"学术词汇类型: {stats['academic_types']}")
        print(f"学科词汇类型: {stats['discipline_types']}")
        
        # 显示学术分析
        academic = results["academic_analysis"]
        print(f"学术词汇比例: {academic['overall_ratio']*100:.1f}%")
        
        # 显示学科分析
        discipline = results["discipline_analysis"]
        print(f"主要学科: {discipline['primary_discipline']}")
        
        # 显示词表信息
        wordlist_info = analyzer.get_wordlist_info()
        print(f"使用的学术词表: {list(wordlist_info['academic_wordlists'].keys())}")
        print(f"使用的学科词表: {list(wordlist_info['discipline_wordlists'].keys())}")
        
        return True
        
    print(f"增强版分析器测试失败: {results.get('error', '未知错误')}")
    return False


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    if test_enhanced_analyzer():
        print("\n增强版学术词汇分析器测试通过!")
    else:
        print("\n增强版学术词汇分析器测试失败!")
