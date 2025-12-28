#!/usr/bin/env python3
"""
analyzer_base.py - 分析器基类

提供分析器的基本接口和配置管理。
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class AnalysisResult:
    """分析结果数据类"""
    analyzer_name: str
    results: Dict[str, Any]
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "analyzer_name": self.analyzer_name,
            "results": self.results,
            "metadata": self.metadata,
            "error": self.error,
            "success": self.error is None
        }
    
    def is_successful(self) -> bool:
        """检查是否成功"""
        return self.error is None


class BaseAnalyzer(ABC):
    """分析器基类"""
    
    def __init__(self, name: str):
        """
        初始化分析器
        
        Args:
            name: 分析器名称
        """
        self.name = name
        self.logger = logging.getLogger(f"analyzer.{name}")
        
    @abstractmethod
    def analyze(self, texts: List[str], metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        分析文本
        
        Args:
            texts: 文本列表
            metadata: 元数据
            
        Returns:
            分析结果
        """
        pass
    
    def validate_input(self, texts: List[str]) -> bool:
        """
        验证输入
        
        Args:
            texts: 文本列表
            
        Returns:
            是否有效
        """
        if not texts:
            self.logger.warning("输入文本列表为空")
            return False
            
        if not all(isinstance(text, str) for text in texts):
            self.logger.warning("输入包含非字符串元素")
            return False
            
        return True
    
    def create_error_result(self, error_message: str, metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        创建错误结果
        
        Args:
            error_message: 错误信息
            metadata: 元数据
            
        Returns:
            错误结果
        """
        if metadata is None:
            metadata = {}
            
        return AnalysisResult(
            analyzer_name=self.name,
            results={},
            metadata=metadata,
            error=error_message
        )
    
    def create_success_result(self, results: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        创建成功结果
        
        Args:
            results: 分析结果
            metadata: 元数据
            
        Returns:
            成功结果
        """
        if metadata is None:
            metadata = {}
            
        return AnalysisResult(
            analyzer_name=self.name,
            results=results,
            metadata=metadata,
            error=None
        )


class ConfigurableAnalyzer(BaseAnalyzer):
    """可配置的分析器基类"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化可配置分析器
        
        Args:
            name: 分析器名称
            config: 配置字典
        """
        super().__init__(name)
        self.config = config or {}
        self._validate_config()
        
    def _validate_config(self):
        """验证配置"""
        # 子类可以重写此方法
        pass
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return self.config.get(key, default)
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        更新配置
        
        Args:
            new_config: 新配置
        """
        self.config.update(new_config)
        self._validate_config()
        self.logger.info(f"配置已更新: {list(new_config.keys())}")


class BatchAnalyzer(BaseAnalyzer):
    """批量分析器基类"""
    
    def __init__(self, name: str, batch_size: int = 1000):
        """
        初始化批量分析器
        
        Args:
            name: 分析器名称
            batch_size: 批处理大小
        """
        super().__init__(name)
        self.batch_size = batch_size
    
    def analyze_batch(self, texts: List[str], metadata: Optional[Dict[str, Any]] = None) -> List[AnalysisResult]:
        """
        批量分析文本
        
        Args:
            texts: 文本列表
            metadata: 元数据
            
        Returns:
            分析结果列表
        """
        if not self.validate_input(texts):
            return [self.create_error_result("输入无效", metadata)]
            
        results = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            self.logger.info(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 个文本)")
            
            try:
                batch_metadata = metadata.copy() if metadata else {}
                batch_metadata.update({
                    "batch_number": batch_num,
                    "total_batches": total_batches,
                    "batch_size": len(batch)
                })
                
                result = self.analyze(batch, batch_metadata)
                results.append(result)
                
            except Exception as e:
                error_msg = f"批次 {batch_num} 分析失败: {str(e)}"
                self.logger.error(error_msg)
                results.append(self.create_error_result(error_msg, metadata))
                
        return results
    
    def merge_results(self, results: List[AnalysisResult]) -> AnalysisResult:
        """
        合并多个分析结果
        
        Args:
            results: 分析结果列表
            
        Returns:
            合并后的分析结果
        """
        if not results:
            return self.create_error_result("没有可合并的结果")
            
        # 检查是否有错误
        errors = [r.error for r in results if r.error]
        if errors:
            error_msg = f"合并时发现错误: {', '.join(errors[:3])}"
            if len(errors) > 3:
                error_msg += f" ... (共 {len(errors)} 个错误)"
            return self.create_error_result(error_msg)
            
        # 合并成功的结果
        merged_results = {}
        merged_metadata = {}
        
        for result in results:
            merged_results.update(result.results)
            merged_metadata.update(result.metadata)
            
        return self.create_success_result(merged_results, merged_metadata)


def save_results_to_file(results: Union[AnalysisResult, List[AnalysisResult]], 
                        output_file: Path):
    """
    保存分析结果到文件
    
    Args:
        results: 分析结果或列表
        output_file: 输出文件路径
    """
    if isinstance(results, AnalysisResult):
        data = [results.to_dict()]
    else:
        data = [r.to_dict() for r in results]
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_results_from_file(input_file: Path) -> List[AnalysisResult]:
    """
    从文件加载分析结果
    
    Args:
        input_file: 输入文件路径
        
    Returns:
        分析结果列表
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = []
    for item in data:
        result = AnalysisResult(
            analyzer_name=item.get("analyzer_name", "unknown"),
            results=item.get("results", {}),
            metadata=item.get("metadata", {}),
            error=item.get("error")
        )
        results.append(result)
        
    return results
