#!/usr/bin/env python3
"""
StatisticCorpus.py - 统计corpus中不同版本json文件的基本信息

该程序读取corpus/v0.0/和corpus/v1.0/目录下的所有json文件，
提取基本信息如：文件数量、文章数量、字段结构、数据类型等统计信息。
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set, Tuple
import datetime


class CorpusStatistic:
    """统计corpus信息的类"""
    
    def __init__(self, corpus_root: str = "corpus"):
        """
        初始化统计器
        
        Args:
            corpus_root: corpus根目录路径
        """
        self.corpus_root = Path(corpus_root)
        self.stats = {
            "versions": {},
            "overall": {
                "total_files": 0,
                "total_articles": 0,
                "unique_topics": set(),
                "unique_sections": set(),
                "date_range": {"min": None, "max": None}
            }
        }
        
    def analyze_version(self, version: str) -> Dict[str, Any]:
        """
        分析特定版本的corpus
        
        Args:
            version: 版本号，如 "v0.0" 或 "v1.0"
            
        Returns:
            版本统计信息字典
        """
        version_path = self.corpus_root / version / "data"
        if not version_path.exists():
            print(f"警告: 版本 {version} 的数据目录不存在: {version_path}")
            return {}
            
        json_files = list(version_path.glob("*.json"))
        if not json_files:
            print(f"警告: 版本 {version} 没有找到json文件")
            return {}
            
        version_stats = {
            "version": version,
            "file_count": len(json_files),
            "article_count": 0,
            "files": [],
            "field_stats": defaultdict(lambda: {"count": 0, "types": set(), "sample_values": []}),
            "topic_distribution": Counter(),
            "section_distribution": Counter(),
            "author_stats": Counter(),
            "date_stats": {"min": None, "max": None, "count": 0},
            "text_stats": {"total_length": 0, "avg_length": 0, "min_length": float('inf'), "max_length": 0},
            "unique_fields": set()
        }
        
        for json_file in json_files:
            file_stats = self._analyze_json_file(json_file, version)
            version_stats["files"].append(file_stats)
            
            # 更新版本级统计
            version_stats["article_count"] += file_stats["article_count"]
            version_stats["topic_distribution"].update(file_stats["topic_distribution"])
            version_stats["section_distribution"].update(file_stats["section_distribution"])
            version_stats["author_stats"].update(file_stats["author_stats"])
            
            # 更新字段统计
            for field, field_info in file_stats["field_stats"].items():
                version_stats["field_stats"][field]["count"] += field_info["count"]
                version_stats["field_stats"][field]["types"].update(field_info["types"])
                # 只保留前3个样本值
                if len(version_stats["field_stats"][field]["sample_values"]) < 3:
                    version_stats["field_stats"][field]["sample_values"].extend(
                        field_info["sample_values"][:3 - len(version_stats["field_stats"][field]["sample_values"])]
                    )
                    
            # 更新唯一字段集合
            version_stats["unique_fields"].update(file_stats["unique_fields"])
            
            # 更新日期统计
            if file_stats["date_stats"]["min"]:
                if version_stats["date_stats"]["min"] is None or file_stats["date_stats"]["min"] < version_stats["date_stats"]["min"]:
                    version_stats["date_stats"]["min"] = file_stats["date_stats"]["min"]
                if version_stats["date_stats"]["max"] is None or file_stats["date_stats"]["max"] > version_stats["date_stats"]["max"]:
                    version_stats["date_stats"]["max"] = file_stats["date_stats"]["max"]
                version_stats["date_stats"]["count"] += file_stats["date_stats"]["count"]
                
            # 更新文本统计
            version_stats["text_stats"]["total_length"] += file_stats["text_stats"]["total_length"]
            version_stats["text_stats"]["min_length"] = min(
                version_stats["text_stats"]["min_length"], 
                file_stats["text_stats"]["min_length"]
            )
            version_stats["text_stats"]["max_length"] = max(
                version_stats["text_stats"]["max_length"], 
                file_stats["text_stats"]["max_length"]
            )
            
        # 计算平均文本长度
        if version_stats["article_count"] > 0:
            version_stats["text_stats"]["avg_length"] = (
                version_stats["text_stats"]["total_length"] / version_stats["article_count"]
            )
            
        # 转换set为list以便JSON序列化
        for field in version_stats["field_stats"]:
            version_stats["field_stats"][field]["types"] = list(version_stats["field_stats"][field]["types"])
            
        version_stats["unique_fields"] = list(version_stats["unique_fields"])
        
        return version_stats
    
    def _analyze_json_file(self, json_file: Path, version: str) -> Dict[str, Any]:
        """
        分析单个json文件
        
        Args:
            json_file: json文件路径
            version: 版本号
            
        Returns:
            文件统计信息字典
        """
        file_stats = {
            "filename": json_file.name,
            "filepath": str(json_file),
            "article_count": 0,
            "field_stats": defaultdict(lambda: {"count": 0, "types": set(), "sample_values": []}),
            "topic_distribution": Counter(),
            "section_distribution": Counter(),
            "author_stats": Counter(),
            "date_stats": {"min": None, "max": None, "count": 0},
            "text_stats": {"total_length": 0, "avg_length": 0, "min_length": float('inf'), "max_length": 0},
            "unique_fields": set()
        }
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                print(f"警告: {json_file} 不是JSON数组")
                return file_stats
                
            file_stats["article_count"] = len(data)
            
            for article in data:
                if not isinstance(article, dict):
                    continue
                    
                # 分析字段
                for field, value in article.items():
                    file_stats["unique_fields"].add(field)
                    
                    field_info = file_stats["field_stats"][field]
                    field_info["count"] += 1
                    field_info["types"].add(type(value).__name__)
                    
                    # 收集样本值（只收集前3个不同的值）
                    if len(field_info["sample_values"]) < 3 and value not in field_info["sample_values"]:
                        # 对于长文本，只取前100字符
                        if isinstance(value, str) and len(value) > 100:
                            field_info["sample_values"].append(value[:100] + "...")
                        elif isinstance(value, list) and len(value) > 3:
                            field_info["sample_values"].append(f"列表长度: {len(value)}")
                        else:
                            field_info["sample_values"].append(value)
                
                # 特定字段的统计
                # 主题分布
                if "topic" in article and article["topic"]:
                    file_stats["topic_distribution"][article["topic"]] += 1
                    
                # 部分分布
                if "section" in article and article["section"]:
                    file_stats["section_distribution"][article["section"]] += 1
                    
                # 作者统计
                if "authors" in article and isinstance(article["authors"], list):
                    for author in article["authors"]:
                        file_stats["author_stats"][author] += 1
                        
                # 日期统计
                if "published_time" in article and article["published_time"]:
                    try:
                        # 尝试解析ISO格式日期
                        date_str = article["published_time"]
                        # 简化处理，只取日期部分
                        if "T" in date_str:
                            date_part = date_str.split("T")[0]
                            date_obj = datetime.datetime.strptime(date_part, "%Y-%m-%d").date()
                            
                            if file_stats["date_stats"]["min"] is None or date_obj < file_stats["date_stats"]["min"]:
                                file_stats["date_stats"]["min"] = date_obj
                            if file_stats["date_stats"]["max"] is None or date_obj > file_stats["date_stats"]["max"]:
                                file_stats["date_stats"]["max"] = date_obj
                                
                            file_stats["date_stats"]["count"] += 1
                    except (ValueError, AttributeError):
                        pass
                        
                # 文本统计
                if "text" in article and isinstance(article["text"], list):
                    text_length = sum(len(str(item)) for item in article["text"])
                    file_stats["text_stats"]["total_length"] += text_length
                    file_stats["text_stats"]["min_length"] = min(
                        file_stats["text_stats"]["min_length"], text_length
                    )
                    file_stats["text_stats"]["max_length"] = max(
                        file_stats["text_stats"]["max_length"], text_length
                    )
                    
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"错误: 无法解析文件 {json_file}: {e}")
            return file_stats
            
        # 计算平均文本长度
        if file_stats["article_count"] > 0:
            file_stats["text_stats"]["avg_length"] = (
                file_stats["text_stats"]["total_length"] / file_stats["article_count"]
            )
            
        # 转换set为list
        for field in file_stats["field_stats"]:
            file_stats["field_stats"][field]["types"] = list(file_stats["field_stats"][field]["types"])
            
        file_stats["unique_fields"] = list(file_stats["unique_fields"])
        
        return file_stats
    
    def analyze_all(self) -> Dict[str, Any]:
        """
        分析所有版本的corpus
        
        Returns:
            完整的统计信息字典
        """
        # 检测所有版本
        versions = []
        for item in self.corpus_root.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                versions.append(item.name)
                
        versions.sort()  # 按版本排序
        
        print(f"找到 {len(versions)} 个版本: {versions}")
        
        for version in versions:
            print(f"\n分析版本: {version}")
            version_stats = self.analyze_version(version)
            if version_stats:
                self.stats["versions"][version] = version_stats
                
                # 更新总体统计
                self.stats["overall"]["total_files"] += version_stats["file_count"]
                self.stats["overall"]["total_articles"] += version_stats["article_count"]
                
                # 更新唯一主题和部分
                for topic in version_stats["topic_distribution"]:
                    self.stats["overall"]["unique_topics"].add(topic)
                for section in version_stats["section_distribution"]:
                    self.stats["overall"]["unique_sections"].add(section)
                    
                # 更新日期范围
                if version_stats["date_stats"]["min"]:
                    if self.stats["overall"]["date_range"]["min"] is None or version_stats["date_stats"]["min"] < self.stats["overall"]["date_range"]["min"]:
                        self.stats["overall"]["date_range"]["min"] = version_stats["date_stats"]["min"]
                    if self.stats["overall"]["date_range"]["max"] is None or version_stats["date_stats"]["max"] > self.stats["overall"]["date_range"]["max"]:
                        self.stats["overall"]["date_range"]["max"] = version_stats["date_stats"]["max"]
                        
        # 转换set为list
        self.stats["overall"]["unique_topics"] = list(self.stats["overall"]["unique_topics"])
        self.stats["overall"]["unique_sections"] = list(self.stats["overall"]["unique_sections"])
        
        return self.stats
    
    def print_summary(self, stats: Dict[str, Any] = None):
        """
        打印统计摘要
        
        Args:
            stats: 统计信息字典，如果为None则使用self.stats
        """
        if stats is None:
            stats = self.stats
            
        print("=" * 80)
        print("CORPUS 统计摘要")
        print("=" * 80)
        
        print(f"\n总体统计:")
        print(f"  版本数量: {len(stats['versions'])}")
        print(f"  文件总数: {stats['overall']['total_files']}")
        print(f"  文章总数: {stats['overall']['total_articles']}")
        print(f"  唯一主题数: {len(stats['overall']['unique_topics'])}")
        print(f"  唯一部分数: {len(stats['overall']['unique_sections'])}")
        
        if stats['overall']['date_range']['min'] and stats['overall']['date_range']['max']:
            print(f"  日期范围: {stats['overall']['date_range']['min']} 到 {stats['overall']['date_range']['max']}")
            
        for version, version_stats in stats["versions"].items():
            print(f"\n版本 {version}:")
            print(f"  文件数: {version_stats['file_count']}")
            print(f"  文章数: {version_stats['article_count']}")
            print(f"  平均每文件文章数: {version_stats['article_count'] / version_stats['file_count']:.1f}")
            print(f"  唯一字段数: {len(version_stats['unique_fields'])}")
            
            # 显示前5个最常见的主题
            if version_stats['topic_distribution']:
                print(f"  主题分布 (前5):")
                for topic, count in version_stats['topic_distribution'].most_common(5):
                    print(f"    - {topic}: {count} 篇文章")
                    
            # 显示字段统计摘要
            print(f"  字段统计:")
            for field, field_info in sorted(version_stats['field_stats'].items()):
                presence = (field_info['count'] / version_stats['article_count'] * 100) if version_stats['article_count'] > 0 else 0
                print(f"    - {field}: {presence:.1f}% 出现率, 类型: {', '.join(field_info['types'])}")
                
            # 文本统计
            print(f"  文本统计:")
            print(f"    - 总字符数: {version_stats['text_stats']['total_length']:,}")
            print(f"    - 平均长度: {version_stats['text_stats']['avg_length']:.0f} 字符")
            print(f"    - 最小长度: {version_stats['text_stats']['min_length']:,} 字符")
            print(f"    - 最大长度: {version_stats['text_stats']['max_length']:,} 字符")
            
    def save_to_json(self, output_file: str = "corpus_statistics.json"):
        """
        将统计结果保存为JSON文件
        
        Args:
            output_file: 输出文件路径
        """
        # 准备可序列化的数据
        serializable_stats = self._make_serializable(self.stats)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"\n统计结果已保存到: {output_file}")
        
    def _make_serializable(self, obj: Any) -> Any:
        """
        将对象转换为可JSON序列化的格式
        
        Args:
            obj: 任意对象
            
        Returns:
            可序列化的对象
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        elif isinstance(obj, Counter):
            return dict(obj)
        elif isinstance(obj, defaultdict):
            return dict(obj)
        else:
            return str(obj)


def main():
    """主函数"""
    print("开始分析corpus...")
    
    # 创建统计器
    statistic = CorpusStatistic("corpus")
    
    # 分析所有版本
    stats = statistic.analyze_all()
    
    # 打印摘要
    statistic.print_summary(stats)
    
    # 保存到JSON文件
    statistic.save_to_json("corpus_statistics.json")
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
