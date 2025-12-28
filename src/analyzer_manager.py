#!/usr/bin/env python3
"""
analyzer_manager.py - 分析器管理器

管理多个分析器，支持流水线操作和结果聚合。
"""

import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass

from analyzer_base import BaseAnalyzer, AnalysisResult


@dataclass
class PipelineConfig:
    """流水线配置"""
    name: str
    analyzer_names: List[str]
    description: str = ""
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class AnalyzerManager:
    """分析器管理器"""
    
    def __init__(self, name: str):
        """
        初始化分析器管理器
        
        Args:
            name: 管理器名称
        """
        self.name = name
        self.logger = logging.getLogger(f"analyzer_manager.{name}")
        
        # 注册的分析器
        self.analyzers: Dict[str, BaseAnalyzer] = {}
        
        # 流水线配置
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.active_pipeline: Optional[str] = None
        
    def register_analyzer(self, analyzer: BaseAnalyzer) -> bool:
        """
        注册分析器
        
        Args:
            analyzer: 分析器实例
            
        Returns:
            是否成功
        """
        if analyzer.name in self.analyzers:
            self.logger.warning(f"分析器 '{analyzer.name}' 已存在，将被覆盖")
            
        self.analyzers[analyzer.name] = analyzer
        self.logger.info(f"注册分析器: {analyzer.name}")
        return True
        
    def unregister_analyzer(self, analyzer_name: str) -> bool:
        """
        注销分析器
        
        Args:
            analyzer_name: 分析器名称
            
        Returns:
            是否成功
        """
        if analyzer_name not in self.analyzers:
            self.logger.warning(f"分析器 '{analyzer_name}' 不存在")
            return False
            
        del self.analyzers[analyzer_name]
        self.logger.info(f"注销分析器: {analyzer_name}")
        return True
        
    def get_analyzer(self, analyzer_name: str) -> Optional[BaseAnalyzer]:
        """
        获取分析器
        
        Args:
            analyzer_name: 分析器名称
            
        Returns:
            分析器实例，如果不存在则返回None
        """
        return self.analyzers.get(analyzer_name)
        
    def list_analyzers(self) -> List[str]:
        """
        列出所有分析器名称
        
        Returns:
            分析器名称列表
        """
        return list(self.analyzers.keys())
        
    def create_pipeline(self, name: str, analyzer_names: List[str], 
                       description: str = "", parameters: Dict[str, Any] = None) -> bool:
        """
        创建流水线
        
        Args:
            name: 流水线名称
            analyzer_names: 分析器名称列表
            description: 描述
            parameters: 参数
            
        Returns:
            是否成功
        """
        # 验证分析器是否存在
        missing_analyzers = [name for name in analyzer_names if name not in self.analyzers]
        if missing_analyzers:
            self.logger.error(f"创建流水线失败，缺少分析器: {missing_analyzers}")
            return False
            
        pipeline = PipelineConfig(
            name=name,
            analyzer_names=analyzer_names,
            description=description,
            parameters=parameters or {}
        )
        
        self.pipelines[name] = pipeline
        self.logger.info(f"创建流水线 '{name}'，包含 {len(analyzer_names)} 个分析器")
        return True
        
    def delete_pipeline(self, pipeline_name: str) -> bool:
        """
        删除流水线
        
        Args:
            pipeline_name: 流水线名称
            
        Returns:
            是否成功
        """
        if pipeline_name not in self.pipelines:
            self.logger.warning(f"流水线 '{pipeline_name}' 不存在")
            return False
            
        del self.pipelines[pipeline_name]
        
        # 如果删除的是活动流水线，清空活动流水线
        if self.active_pipeline == pipeline_name:
            self.active_pipeline = None
            
        self.logger.info(f"删除流水线: {pipeline_name}")
        return True
        
    def set_active_pipeline(self, pipeline_name: str) -> bool:
        """
        设置活动流水线
        
        Args:
            pipeline_name: 流水线名称
            
        Returns:
            是否成功
        """
        if pipeline_name not in self.pipelines:
            self.logger.error(f"流水线 '{pipeline_name}' 不存在")
            return False
            
        self.active_pipeline = pipeline_name
        self.logger.info(f"设置活动流水线: {pipeline_name}")
        return True
        
    def get_active_pipeline(self) -> Optional[PipelineConfig]:
        """
        获取活动流水线配置
        
        Returns:
            流水线配置，如果没有活动流水线则返回None
        """
        if self.active_pipeline:
            return self.pipelines.get(self.active_pipeline)
        return None
        
    def list_pipelines(self) -> List[str]:
        """
        列出所有流水线名称
        
        Returns:
            流水线名称列表
        """
        return list(self.pipelines.keys())
        
    def run_pipeline(self, texts: List[str], metadata: Optional[Dict[str, Any]] = None,
                    pipeline_name: Optional[str] = None) -> Dict[str, AnalysisResult]:
        """
        运行流水线
        
        Args:
            texts: 文本列表
            metadata: 元数据
            pipeline_name: 流水线名称，如果为None则使用活动流水线
            
        Returns:
            分析结果字典，键为分析器名称
        """
        # 确定要运行的流水线
        if pipeline_name is None:
            if self.active_pipeline is None:
                self.logger.error("没有指定流水线且没有活动流水线")
                return {}
            pipeline_name = self.active_pipeline
            
        if pipeline_name not in self.pipelines:
            self.logger.error(f"流水线 '{pipeline_name}' 不存在")
            return {}
            
        pipeline = self.pipelines[pipeline_name]
        self.logger.info(f"运行流水线 '{pipeline_name}'，包含 {len(pipeline.analyzer_names)} 个分析器")
        
        results = {}
        
        for analyzer_name in pipeline.analyzer_names:
            analyzer = self.analyzers.get(analyzer_name)
            if not analyzer:
                self.logger.error(f"分析器 '{analyzer_name}' 不存在")
                continue
                
            self.logger.info(f"运行分析器: {analyzer_name}")
            
            try:
                # 合并流水线参数和元数据
                analyzer_metadata = metadata.copy() if metadata else {}
                analyzer_metadata.update({
                    "pipeline_name": pipeline_name,
                    "analyzer_name": analyzer_name
                })
                
                # 运行分析器
                result = analyzer.analyze(texts, analyzer_metadata)
                results[analyzer_name] = result
                
                if result.error:
                    self.logger.error(f"分析器 '{analyzer_name}' 失败: {result.error}")
                else:
                    self.logger.info(f"分析器 '{analyzer_name}' 完成")
                    
            except Exception as e:
                error_msg = f"分析器 '{analyzer_name}' 运行时异常: {str(e)}"
                self.logger.error(error_msg)
                results[analyzer_name] = analyzer.create_error_result(error_msg, metadata)
                
        return results
        
    def run_analyzer(self, analyzer_name: str, texts: List[str], 
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[AnalysisResult]:
        """
        运行单个分析器
        
        Args:
            analyzer_name: 分析器名称
            texts: 文本列表
            metadata: 元数据
            
        Returns:
            分析结果，如果分析器不存在则返回None
        """
        analyzer = self.analyzers.get(analyzer_name)
        if not analyzer:
            self.logger.error(f"分析器 '{analyzer_name}' 不存在")
            return None
            
        self.logger.info(f"运行分析器: {analyzer_name}")
        
        try:
            result = analyzer.analyze(texts, metadata)
            
            if result.error:
                self.logger.error(f"分析器 '{analyzer_name}' 失败: {result.error}")
            else:
                self.logger.info(f"分析器 '{analyzer_name}' 完成")
                
            return result
            
        except Exception as e:
            error_msg = f"分析器 '{analyzer_name}' 运行时异常: {str(e)}"
            self.logger.error(error_msg)
            return analyzer.create_error_result(error_msg, metadata)
            
    def aggregate_results(self, results: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """
        聚合多个分析器的结果
        
        Args:
            results: 分析结果字典
            
        Returns:
            聚合结果
        """
        aggregated = {
            "total_analyzers": len(results),
            "successful_analyzers": 0,
            "failed_analyzers": 0,
            "results": {},
            "errors": []
        }
        
        for analyzer_name, result in results.items():
            if result.is_successful():
                aggregated["successful_analyzers"] += 1
                aggregated["results"][analyzer_name] = result.to_dict()
            else:
                aggregated["failed_analyzers"] += 1
                aggregated["errors"].append({
                    "analyzer": analyzer_name,
                    "error": result.error
                })
                
        return aggregated
        
    def save_pipeline_config(self, output_file: Path):
        """
        保存流水线配置到文件
        
        Args:
            output_file: 输出文件路径
        """
        import json
        
        config_data = {
            "manager_name": self.name,
            "active_pipeline": self.active_pipeline,
            "analyzers": list(self.analyzers.keys()),
            "pipelines": {}
        }
        
        for name, pipeline in self.pipelines.items():
            config_data["pipelines"][name] = {
                "analyzer_names": pipeline.analyzer_names,
                "description": pipeline.description,
                "parameters": pipeline.parameters
            }
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"流水线配置保存到: {output_file}")
        
    def load_pipeline_config(self, input_file: Path) -> bool:
        """
        从文件加载流水线配置
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            是否成功
        """
        import json
        
        if not input_file.exists():
            self.logger.error(f"配置文件不存在: {input_file}")
            return False
            
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            # 注意：这里只加载配置，不重新创建分析器实例
            # 分析器实例需要先注册
            
            self.active_pipeline = config_data.get("active_pipeline")
            
            # 清空现有流水线
            self.pipelines.clear()
            
            # 加载流水线配置
            pipelines_data = config_data.get("pipelines", {})
            for name, pipeline_data in pipelines_data.items():
                pipeline = PipelineConfig(
                    name=name,
                    analyzer_names=pipeline_data.get("analyzer_names", []),
                    description=pipeline_data.get("description", ""),
                    parameters=pipeline_data.get("parameters", {})
                )
                self.pipelines[name] = pipeline
                
            self.logger.info(f"从文件加载流水线配置: {input_file}")
            self.logger.info(f"加载了 {len(self.pipelines)} 个流水线")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载流水线配置失败: {str(e)}")
            return False
