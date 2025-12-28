#!/usr/bin/env python3
"""
wordlist_downloader.py - 词表下载器

从权威网站下载公认的词表数据，包括学术词表、常用词表和学科词表。
"""

import os
import sys
import json
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging
import time
import hashlib
from urllib.parse import urlparse

# 导入NLTK词表
import nltk
from nltk.corpus import stopwords, words, wordnet


class WordlistDownloader:
    """词表下载器，从权威网站下载词表数据"""
    
    def __init__(self, cache_dir: str = "data/wordlists"):
        """
        初始化词表下载器
        
        Args:
            cache_dir: 词表缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("wordlist_downloader")
        
        # 词表源配置
        self.wordlist_sources = {
            # 学术词表
            "awl": {
                "name": "Academic Word List (AWL)",
                "url": "https://www.wgtn.ac.nz/__data/assets/file/0006/1851492/AWL-headwords.csv",
                "format": "csv",
                "description": "Coxhead Academic Word List - 570个学术词汇"
            },
            # 常用词表（备用源）
            "coca_top_5000": {
                "name": "COCA Top 5000 Words",
                "url": "https://www.wordfrequency.info/samples/lemma5000.csv",
                "format": "csv",
                "description": "COCA语料库前5000个最常用词"
            },
            # 学科词表（示例）
            "science_terms": {
                "name": "Science Terminology",
                "url": "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
                "format": "txt",
                "description": "通用英语词汇表（包含科学术语）"
            }
        }
        
        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def download_wordlist(self, wordlist_id: str, force_download: bool = False) -> Optional[Path]:
        """
        下载词表
        
        Args:
            wordlist_id: 词表ID
            force_download: 是否强制重新下载
            
        Returns:
            本地文件路径，如果下载失败则返回None
        """
        if wordlist_id not in self.wordlist_sources:
            self.logger.error(f"未知的词表ID: {wordlist_id}")
            return None
            
        source = self.wordlist_sources[wordlist_id]
        url = source["url"]
        format = source.get("format", "txt")
        
        # 生成缓存文件名
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"{wordlist_id}_{url_hash}.{format}"
        
        # 检查缓存
        if cache_file.exists() and not force_download:
            self.logger.info(f"使用缓存的词表: {wordlist_id} -> {cache_file}")
            return cache_file
            
        self.logger.info(f"下载词表: {source['name']} from {url}")
        
        try:
            # 下载文件
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # 保存到缓存
            with open(cache_file, 'wb') as f:
                f.write(response.content)
                
            self.logger.info(f"词表下载成功: {cache_file} ({len(response.content)} bytes)")
            return cache_file
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"下载词表失败 {wordlist_id}: {e}")
            return None
            
    def load_nltk_wordlists(self) -> Dict[str, Set[str]]:
        """
        加载NLTK内置词表
        
        Returns:
            词表字典
        """
        self.logger.info("加载NLTK词表...")
        
        nltk_wordlists = {}
        
        try:
            # 停用词表
            nltk_stopwords = set(stopwords.words('english'))
            nltk_wordlists['nltk_stopwords'] = nltk_stopwords
            self.logger.info(f"加载NLTK停用词: {len(nltk_stopwords)} 个词")
            
            # 英语词汇表
            nltk_words = set(words.words())
            nltk_wordlists['nltk_words'] = nltk_words
            self.logger.info(f"加载NLTK英语词汇: {len(nltk_words)} 个词")
            
            # WordNet词汇
            wordnet_words = set()
            for synset in wordnet.all_synsets():
                for lemma in synset.lemmas():
                    wordnet_words.add(lemma.name().replace('_', ' '))
            nltk_wordlists['wordnet'] = wordnet_words
            self.logger.info(f"加载WordNet词汇: {len(wordnet_words)} 个词")
            
        except Exception as e:
            self.logger.error(f"加载NLTK词表失败: {e}")
            
        return nltk_wordlists
        
    def parse_wordlist_file(self, file_path: Path, format: str = "auto") -> List[str]:
        """
        解析词表文件
        
        Args:
            file_path: 文件路径
            format: 文件格式 (csv, txt, json, auto)
            
        Returns:
            词表列表
        """
        if not file_path.exists():
            self.logger.error(f"词表文件不存在: {file_path}")
            return []
            
        # 自动检测格式
        if format == "auto":
            if file_path.suffix.lower() == '.csv':
                format = 'csv'
            elif file_path.suffix.lower() == '.json':
                format = 'json'
            else:
                format = 'txt'
                
        try:
            if format == 'csv':
                # 读取CSV文件
                df = pd.read_csv(file_path)
                words = []
                
                # 尝试不同的列名
                possible_columns = ['word', 'lemma', 'headword', 'term', 'vocabulary']
                for col in possible_columns:
                    if col in df.columns:
                        words = df[col].dropna().astype(str).str.lower().tolist()
                        break
                        
                if not words and len(df.columns) > 0:
                    # 使用第一列
                    words = df.iloc[:, 0].dropna().astype(str).str.lower().tolist()
                    
                return words
                
            elif format == 'txt':
                # 读取文本文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # 清理和过滤
                words = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 可能包含频率信息，如 "the 12345"
                        parts = line.split()
                        if parts:
                            words.append(parts[0].lower())
                            
                return words
                
            elif format == 'json':
                # 读取JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    return [str(item).lower() for item in data if item]
                elif isinstance(data, dict):
                    # 可能是词频字典
                    return list(data.keys())
                else:
                    return []
                    
            else:
                self.logger.error(f"不支持的格式: {format}")
                return []
                
        except Exception as e:
            self.logger.error(f"解析词表文件失败 {file_path}: {e}")
            return []
            
    def download_all_wordlists(self, force_download: bool = False) -> Dict[str, Any]:
        """
        下载所有词表
        
        Args:
            force_download: 是否强制重新下载
            
        Returns:
            下载结果字典
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "downloaded": {},
            "failed": [],
            "nltk_loaded": False
        }
        
        # 下载配置的词表
        for wordlist_id in self.wordlist_sources:
            self.logger.info(f"处理词表: {wordlist_id}")
            
            file_path = self.download_wordlist(wordlist_id, force_download)
            if file_path:
                # 解析词表
                words = self.parse_wordlist_file(file_path)
                
                results["downloaded"][wordlist_id] = {
                    "name": self.wordlist_sources[wordlist_id]["name"],
                    "file_path": str(file_path),
                    "word_count": len(words),
                    "sample_words": words[:10] if words else []
                }
                
                self.logger.info(f"词表 {wordlist_id}: {len(words)} 个词")
            else:
                results["failed"].append(wordlist_id)
                self.logger.warning(f"词表下载失败: {wordlist_id}")
                
        # 加载NLTK词表
        nltk_wordlists = self.load_nltk_wordlists()
        if nltk_wordlists:
            results["nltk_wordlists"] = {
                name: len(words) for name, words in nltk_wordlists.items()
            }
            results["nltk_loaded"] = True
            
        # 保存下载结果
        results_file = self.cache_dir / "download_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"词表下载完成，结果保存到: {results_file}")
        
        return results
        
    def get_wordlist_stats(self) -> Dict[str, Any]:
        """
        获取词表统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "cache_dir": str(self.cache_dir),
            "total_files": 0,
            "wordlists": [],
            "timestamp": datetime.now().isoformat()
        }
        
        if not self.cache_dir.exists():
            return stats
            
        # 统计缓存文件
        for file_path in self.cache_dir.glob("*.*"):
            if file_path.is_file():
                stats["total_files"] += 1
                
                # 获取文件信息
                file_stat = {
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "format": file_path.suffix.lower()[1:] if file_path.suffix else "unknown"
                }
                
                # 尝试解析词表大小
                try:
                    words = self.parse_wordlist_file(file_path)
                    file_stat["word_count"] = len(words)
                except:
                    file_stat["word_count"] = 0
                    
                stats["wordlists"].append(file_stat)
                
        return stats
        
    def create_discipline_wordlists(self, corpus_path: str = "corpus/v1.0/data") -> Dict[str, List[str]]:
        """
        从语料库创建学科词表（基于现有语料库）
        
        Args:
            corpus_path: 语料库路径
            
        Returns:
            学科词表字典
        """
        self.logger.info(f"从语料库创建学科词表: {corpus_path}")
        
        discipline_wordlists = {}
        corpus_dir = Path(corpus_path)
        
        if not corpus_dir.exists():
            self.logger.error(f"语料库目录不存在: {corpus_dir}")
            return discipline_wordlists
            
        # 处理每个学科文件
        for file_path in corpus_dir.glob("*.json"):
            discipline = file_path.stem.split('_')[0]  # 提取学科名称
            if discipline.lower() in ["example"]:
                continue
                
            self.logger.info(f"处理学科: {discipline}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 提取文本并分词
                all_words = set()
                for item in data:
                    if isinstance(item, dict):
                        # 提取文本字段
                        text_fields = ["text", "content", "body", "abstract", "title"]
                        for field in text_fields:
                            if field in item and isinstance(item[field], str):
                                words = item[field].lower().split()
                                all_words.update(words)
                                
                # 过滤：只保留字母数字词，长度3-20
                filtered_words = {
                    word for word in all_words 
                    if word.isalpha() and 3 <= len(word) <= 20
                }
                
                discipline_wordlists[discipline] = list(filtered_words)
                self.logger.info(f"学科 {discipline}: {len(filtered_words)} 个词")
                
                # 保存学科词表
                discipline_file = self.cache_dir / f"discipline_{discipline}.txt"
                with open(discipline_file, 'w', encoding='utf-8') as f:
                    for word in sorted(filtered_words):
                        f.write(f"{word}\n")
                        
            except Exception as e:
                self.logger.error(f"处理学科文件失败 {file_path}: {e}")
                
        return discipline_wordlists


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="词表下载器")
    parser.add_argument("--download-all", action="store_true", help="下载所有词表")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--stats", action="store_true", help="显示词表统计")
    parser.add_argument("--create-discipline", action="store_true", 
                       help="从语料库创建学科词表")
    parser.add_argument("--cache-dir", default="data/wordlists", 
                       help="缓存目录路径")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建下载器
    downloader = WordlistDownloader(args.cache_dir)
    
    if args.download_all:
        print("开始下载所有词表...")
        results = downloader.download_all_wordlists(args.force)
        
        print(f"\n下载完成:")
        print(f"  成功: {len(results['downloaded'])} 个词表")
        print(f"  失败: {len(results['failed'])} 个词表")
        
        if results['downloaded']:
            print("\n下载的词表:")
            for wordlist_id, info in results['downloaded'].items():
                print(f"  {wordlist_id}: {info['name']} ({info['word_count']} 词)")
                
    if args.stats:
        print("\n词表统计:")
        stats = downloader.get_wordlist_stats()
        print(f"  缓存目录: {stats['cache_dir']}")
        print(f"  文件总数: {stats['total_files']}")
        
        if stats['wordlists']:
            print("\n  词表文件:")
            for file_info in stats['wordlists']:
                print(f"    {file_info['name']}: {file_info['word_count']} 词, {file_info['size']} bytes")
                
    if args.create_discipline:
        print("\n从语料库创建学科词表...")
        discipline_wordlists = downloader.create_discipline_wordlists()
        
        print(f"\n创建的学科词表:")
        for discipline, words in discipline_wordlists.items():
            print(f"  {discipline}: {len(words)} 个词")
            
    if not any([args.download_all, args.stats, args.create_discipline]):
        print("请指定操作，使用 --help 查看帮助")


if __name__ == "__main__":
    main()
