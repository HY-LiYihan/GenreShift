#!/usr/bin/env python3
"""
enhanced_corpus_reader.py - 增强版语料库读取器

支持：
1. 完整的元数据提取（版本、文件名、学科分类等）
2. 嵌套文本结构的识别和处理
3. 多种分类标准的支持
4. 灵活的文本提取策略
5. 数据标准化输出
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Iterator, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from enum import Enum


class TextExtractionStrategy(Enum):
    """文本提取策略"""
    MAIN_CONTENT_ONLY = "main_content_only"  # 只提取主要内容
    ALL_TEXT_FIELDS = "all_text_fields"      # 提取所有文本字段
    NESTED_STRUCTURE = "nested_structure"    # 处理嵌套结构
    SMART_EXTRACTION = "smart_extraction"    # 智能提取（默认）


class ClassificationField(Enum):
    """分类字段"""
    VERSION = "version"
    DISCIPLINE = "discipline"
    SOURCE_FILE = "source_file"
    FILE_NAME = "file_name"
    YEAR = "year"
    PUBLICATION_TYPE = "publication_type"
    LANGUAGE = "language"
    DOMAIN = "domain"


@dataclass
class EnhancedCorpusDocument:
    """增强版语料库文档表示"""
    
    # 核心标识
    id: str
    text: str
    
    # 分类属性
    version: str = ""
    discipline: str = ""
    source_file: str = ""
    file_name: str = ""
    genre: str = ""  # 新增：体裁字段
    
    # 时间属性
    year: Optional[int] = None
    date: Optional[str] = None
    
    # 内容属性
    title: str = ""
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    
    # 结构属性
    sections: Dict[str, str] = field(default_factory=dict)  # 章节标题 -> 内容
    paragraphs: List[str] = field(default_factory=list)     # 段落列表
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_info: Dict[str, Any] = field(default_factory=dict)
    
    # 处理信息
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    text_length: int = 0
    word_count: int = 0
    
    def __post_init__(self):
        """初始化后计算文本统计"""
        self.text_length = len(self.text)
        self.word_count = len(self.text.split())
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        # 确保所有字段都可序列化
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result
    
    def get_classification_value(self, field: ClassificationField) -> Any:
        """获取分类字段的值"""
        field_map = {
            ClassificationField.VERSION: self.version,
            ClassificationField.DISCIPLINE: self.discipline,
            ClassificationField.SOURCE_FILE: self.source_file,
            ClassificationField.FILE_NAME: self.file_name,
            ClassificationField.YEAR: self.year,
            ClassificationField.PUBLICATION_TYPE: self.metadata.get("publication_type"),
            ClassificationField.LANGUAGE: self.metadata.get("language"),
            ClassificationField.DOMAIN: self.metadata.get("domain"),
        }
        return field_map.get(field)
    
    def get_all_text(self, include_sections: bool = True, include_paragraphs: bool = False) -> str:
        """获取所有文本内容"""
        texts = [self.text]
        
        if include_sections and self.sections:
            texts.extend(self.sections.values())
            
        if include_paragraphs and self.paragraphs:
            texts.extend(self.paragraphs)
            
        return " ".join(texts)


class EnhancedCorpusReader:
    """增强版语料库读取器"""
    
    def __init__(self, corpus_root: str = "corpus", 
                 extraction_strategy: TextExtractionStrategy = TextExtractionStrategy.SMART_EXTRACTION):
        """
        初始化增强版语料库读取器
        
        Args:
            corpus_root: 语料库根目录路径
            extraction_strategy: 文本提取策略
        """
        self.corpus_root = Path(corpus_root)
        self.extraction_strategy = extraction_strategy
        self.logger = logging.getLogger("enhanced_corpus_reader")
        
        # 配置
        self.main_content_fields = ["text", "content", "body", "main_text", "article_body", "full_text"]
        self.secondary_fields = ["abstract", "summary", "title", "description", "keywords"]
        self.metadata_fields = ["year", "date", "authors", "language", "domain", "publication_type", "discipline"]
        
        # 学科映射（从文件名提取）
        self.discipline_mapping = {
            "arts": "Arts & Humanities",
            "humanities": "Arts & Humanities",
            "life": "Life Sciences & Biomedicine",
            "biology": "Life Sciences & Biomedicine",
            "medicine": "Life Sciences & Biomedicine",
            "physical": "Physical Sciences",
            "physics": "Physical Sciences",
            "chemistry": "Physical Sciences",
            "social": "Social Sciences",
            "sociology": "Social Sciences",
            "technology": "Technology",
            "tech": "Technology",
            "engineering": "Technology"
        }
        
        # 缓存
        self._versions: List[str] = []
        self._detect_versions()
        
    def _detect_versions(self):
        """检测可用的语料库版本"""
        if not self.corpus_root.exists():
            self.logger.warning(f"语料库根目录不存在: {self.corpus_root}")
            return
            
        for item in self.corpus_root.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                self._versions.append(item.name)
                
        self._versions.sort()
        self.logger.info(f"检测到 {len(self._versions)} 个版本: {self._versions}")
        
    def list_versions(self) -> List[str]:
        """列出所有可用的版本"""
        return self._versions.copy()
    
    def list_files(self, version: str, pattern: str = "*.json") -> List[Path]:
        """
        列出指定版本的所有数据文件
        
        Args:
            version: 版本号
            pattern: 文件模式
            
        Returns:
            文件路径列表
        """
        version_path = self.corpus_root / version / "data"
        if not version_path.exists():
            self.logger.warning(f"版本 {version} 的数据目录不存在: {version_path}")
            return []
            
        # 获取所有文件，但跳过example.json
        all_files = list(version_path.glob(pattern))
        filtered_files = [f for f in all_files if f.name.lower() != "example.json"]
        
        if len(all_files) != len(filtered_files):
            self.logger.info(f"跳过了 {len(all_files) - len(filtered_files)} 个example.json文件")
            
        return filtered_files
    
    def extract_discipline_from_filename(self, filename: str) -> str:
        """
        从文件名提取学科信息
        
        Args:
            filename: 文件名
            
        Returns:
            学科名称
        """
        filename_lower = filename.lower()
        
        # 检查学科映射
        for keyword, discipline in self.discipline_mapping.items():
            if keyword in filename_lower:
                return discipline
                
        # 尝试从常见模式提取
        patterns = [
            r"([A-Za-z\s&]+)_random",  # 如 "Arts & Humanities_random"
            r"([A-Za-z\s]+)_\w+\.json",  # 如 "Life Sciences_xxx.json"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                discipline = match.group(1).strip()
                return discipline
                
        return "Unknown"
    
    def extract_metadata_from_item(self, item: Dict[str, Any], source_file: Path, version: str) -> Dict[str, Any]:
        """
        从数据项中提取元数据
        
        Args:
            item: 数据项字典
            source_file: 源文件路径
            version: 版本号
            
        Returns:
            元数据字典
        """
        metadata = {}
        
        # 基础元数据
        metadata["source_file"] = str(source_file)
        metadata["file_name"] = source_file.name
        metadata["version"] = version
        
        # 从文件名提取学科
        discipline = self.extract_discipline_from_filename(source_file.name)
        metadata["discipline"] = discipline
        
        # 从数据项中提取元数据字段
        for field in self.metadata_fields:
            if field in item:
                metadata[field] = item[field]
                
        # 尝试提取年份
        if "year" not in metadata and "date" in item:
            date_str = item["date"]
            if isinstance(date_str, str):
                # 尝试从日期字符串中提取年份
                year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                if year_match:
                    metadata["year"] = int(year_match.group())
                    
        # 提取作者信息
        if "authors" in item:
            authors = item["authors"]
            if isinstance(authors, list):
                metadata["authors"] = authors
            elif isinstance(authors, str):
                metadata["authors"] = [author.strip() for author in authors.split(",")]
                
        # 提取关键词
        if "keywords" in item:
            keywords = item["keywords"]
            if isinstance(keywords, list):
                metadata["keywords"] = keywords
            elif isinstance(keywords, str):
                metadata["keywords"] = [kw.strip() for kw in keywords.split(",")]
                
        return metadata
    
    def extract_text_from_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        从数据项中提取文本内容
        
        Args:
            item: 数据项字典
            
        Returns:
            文本内容字典
        """
        result = {
            "main_text": "",
            "title": "",
            "abstract": "",
            "sections": {},
            "paragraphs": [],
            "extraction_method": self.extraction_strategy.value
        }
        
        if self.extraction_strategy == TextExtractionStrategy.MAIN_CONTENT_ONLY:
            result.update(self._extract_main_content_only(item))
        elif self.extraction_strategy == TextExtractionStrategy.ALL_TEXT_FIELDS:
            result.update(self._extract_all_text_fields(item))
        elif self.extraction_strategy == TextExtractionStrategy.NESTED_STRUCTURE:
            result.update(self._extract_nested_structure(item))
        else:  # SMART_EXTRACTION
            result.update(self._extract_smart(item))
            
        return result
    
    def _extract_main_content_only(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """只提取主要内容"""
        result = {}
        
        # 寻找主要内容字段
        for field in self.main_content_fields:
            if field in item:
                content = item[field]
                if isinstance(content, str):
                    result["main_text"] = self._clean_text(content)
                    break
                elif isinstance(content, list):
                    # 如果是列表，连接所有字符串元素
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(self._clean_text(part))
                        elif isinstance(part, dict):
                            # 尝试从字典中提取文本
                            text_parts.append(self._extract_text_from_dict(part))
                    if text_parts:
                        result["main_text"] = " ".join(text_parts)
                        break
                        
        return result
    
    def _extract_all_text_fields(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """提取所有文本字段"""
        result = {}
        all_text_parts = []
        
        # 收集所有文本字段
        for key, value in item.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                if key in self.main_content_fields:
                    all_text_parts.append(self._clean_text(value))
                elif key in self.secondary_fields:
                    if key == "title":
                        result["title"] = self._clean_text(value)
                    elif key == "abstract":
                        result["abstract"] = self._clean_text(value)
                    else:
                        all_text_parts.append(self._clean_text(value))
                elif len(value) > 100:  # 较长的文本字段
                    all_text_parts.append(self._clean_text(value))
                    
        result["main_text"] = " ".join(all_text_parts)
        return result
    
    def _extract_nested_structure(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理嵌套结构"""
        result = {
            "sections": {},
            "paragraphs": []
        }
        
        def process_nested(data, current_section=""):
            if isinstance(data, dict):
                # 检查是否有章节结构
                if "sections" in data and isinstance(data["sections"], list):
                    for section in data["sections"]:
                        if isinstance(section, dict):
                            section_title = section.get("title", "")
                            section_content = section.get("content", "")
                            if section_title and section_content:
                                result["sections"][section_title] = self._clean_text(section_content)
                                
                # 检查是否有段落
                if "paragraphs" in data and isinstance(data["paragraphs"], list):
                    for para in data["paragraphs"]:
                        if isinstance(para, str):
                            result["paragraphs"].append(self._clean_text(para))
                            
                # 递归处理其他字段
                for key, value in data.items():
                    if key not in ["sections", "paragraphs"]:
                        process_nested(value, f"{current_section}.{key}" if current_section else key)
                        
            elif isinstance(data, list):
                for item in data:
                    process_nested(item, current_section)
                    
        process_nested(item)
        
        # 合并所有文本
        all_text = []
        if result["sections"]:
            all_text.extend(result["sections"].values())
        if result["paragraphs"]:
            all_text.extend(result["paragraphs"])
            
        result["main_text"] = " ".join(all_text)
        return result
    
    def _extract_smart(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """智能提取"""
        result = {}
        
        # 首先尝试提取主要内容
        main_content = self._extract_main_content_only(item)
        if main_content.get("main_text"):
            result.update(main_content)
        else:
            # 如果没有找到主要内容，尝试其他方法
            all_text = self._extract_all_text_fields(item)
            result.update(all_text)
            
        # 尝试提取标题和摘要
        for field in ["title", "abstract"]:
            if field in item and isinstance(item[field], str):
                result[field] = self._clean_text(item[field])
                
        return result
    
    def _extract_text_from_dict(self, data: Dict[str, Any]) -> str:
        """从字典中提取文本"""
        text_parts = []
        
        for key, value in data.items():
            if isinstance(value, str):
                text_parts.append(self._clean_text(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        text_parts.append(self._clean_text(item))
                    elif isinstance(item, dict):
                        text_parts.append(self._extract_text_from_dict(item))
                        
        return " ".join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
            
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 移除URL
        text = re.sub(r'https?://\S+', ' ', text)
        
        # 移除特殊字符但保留基本标点
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', ' ', text)
        
        # 标准化空白
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def read_file(self, file_path: Union[str, Path], version: Optional[str] = None) -> List[EnhancedCorpusDocument]:
        """
        读取单个文件
        
        Args:
            file_path: 文件路径
            version: 版本号，如果为None则从路径推断
            
        Returns:
            文档列表
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            self.logger.error(f"文件不存在: {file_path_obj}")
            return []
            
        # 推断版本
        if version is None:
            for v in self._versions:
                if v in str(file_path_obj):
                    version = v
                    break
                    
        if version is None:
            version = "unknown"
            
        try:
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            documents = []
            
            if isinstance(data, list):
                # 处理列表格式
                for i, item in enumerate(data):
                    docs = self._create_documents_from_item(item, file_path_obj, version, i)
                    if docs:
                        documents.extend(docs)
            elif isinstance(data, dict):
                # 处理字典格式（可能包含多个文档）
                if "documents" in data and isinstance(data["documents"], list):
                    # 嵌套的文档列表
                    for i, item in enumerate(data["documents"]):
                        docs = self._create_documents_from_item(item, file_path_obj, version, i)
                        if docs:
                            documents.extend(docs)
                else:
                    # 单个文档
                    docs = self._create_documents_from_item(data, file_path_obj, version, 0)
                    if docs:
                        documents.extend(docs)
                        
            self.logger.info(f"从 {file_path_obj} 读取了 {len(documents)} 个文档")
            return documents
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.error(f"无法解析文件 {file_path_obj}: {e}")
            return []
    
    def _create_documents_from_item(self, item: Dict[str, Any], source_file: Path, 
                                   version: str, index: int) -> List[EnhancedCorpusDocument]:
        """
        从数据项创建多个文档（支持多种体裁）
        
        Args:
            item: 数据项字典
            source_file: 源文件路径
            version: 版本号
            index: 文档索引
            
        Returns:
            文档对象列表
        """
        documents = []
        
        # 提取元数据
        metadata = self.extract_metadata_from_item(item, source_file, version)
        
        # 1. 创建科技新闻文档（来自text字段）
        if "text" in item and item["text"]:
            text_content = item["text"]
            if isinstance(text_content, list):
                # 合并所有段落
                news_text = " ".join([str(p) for p in text_content if p])
            elif isinstance(text_content, str):
                news_text = text_content
            else:
                news_text = str(text_content)
                
            if news_text.strip():
                # 清理文本
                news_text = self._clean_text(news_text)
                
                # 创建文档ID
                doc_id = f"{version}_{source_file.stem}_{index}_news"
                
                # 创建科技新闻文档
                news_doc = EnhancedCorpusDocument(
                    id=doc_id,
                    text=news_text,
                    version=version,
                    discipline=metadata.get("discipline", ""),
                    source_file=str(source_file),
                    file_name=source_file.name,
                    genre="science_news",  # 体裁：科技新闻
                    year=metadata.get("year"),
                    date=metadata.get("date"),
                    title=item.get("title", ""),
                    abstract=item.get("description", ""),  # 使用description作为摘要
                    keywords=metadata.get("keywords", []),
                    authors=metadata.get("authors", []),
                    metadata=metadata,
                    extraction_info={
                        "method": "genre_extraction",
                        "source_field": "text",
                        "genre": "science_news"
                    }
                )
                documents.append(news_doc)
        
        # 2. 创建学术论文文档（来自source_pdf字段中的abstract）
        if "source_pdf" in item and item["source_pdf"]:
            source_pdfs = item["source_pdf"]
            if isinstance(source_pdfs, list):
                for pdf_idx, pdf_item in enumerate(source_pdfs):
                    if isinstance(pdf_item, dict) and "abstract" in pdf_item:
                        abstract_text = pdf_item["abstract"]
                        if abstract_text and isinstance(abstract_text, str):
                            # 清理文本
                            abstract_text = self._clean_text(abstract_text)
                            
                            if abstract_text.strip():
                                # 创建文档ID
                                doc_id = f"{version}_{source_file.stem}_{index}_paper_{pdf_idx}"
                                
                                # 创建学术论文文档
                                paper_doc = EnhancedCorpusDocument(
                                    id=doc_id,
                                    text=abstract_text,
                                    version=version,
                                    discipline=metadata.get("discipline", ""),
                                    source_file=str(source_file),
                                    file_name=source_file.name,
                                    genre="academic_paper",  # 体裁：学术论文
                                    year=metadata.get("year"),
                                    date=metadata.get("date"),
                                    title=pdf_item.get("title", ""),
                                    abstract=abstract_text,  # 摘要就是文本本身
                                    keywords=metadata.get("keywords", []),
                                    authors=pdf_item.get("author", metadata.get("authors", [])),
                                    metadata={
                                        **metadata,
                                        "pdf_title": pdf_item.get("title", ""),
                                        "pdf_authors": pdf_item.get("author", []),
                                        "pdf_index": pdf_idx
                                    },
                                    extraction_info={
                                        "method": "genre_extraction",
                                        "source_field": "source_pdf.abstract",
                                        "genre": "academic_paper",
                                        "pdf_index": pdf_idx
                                    }
                                )
                                documents.append(paper_doc)
        
        # 3. 如果没有提取到任何文档，尝试使用原来的方法
        if not documents:
            original_doc = self._create_document_from_item(item, source_file, version, index)
            if original_doc:
                # 设置默认体裁
                original_doc.genre = "unknown"
                documents.append(original_doc)
        
        return documents
    
    def _create_document_from_item(self, item: Dict[str, Any], source_file: Path, 
                                   version: str, index: int) -> Optional[EnhancedCorpusDocument]:
        """
        从数据项创建文档（旧方法，向后兼容）
        
        Args:
            item: 数据项字典
            source_file: 源文件路径
            version: 版本号
            index: 文档索引
            
        Returns:
            文档对象，如果无法创建则返回None
        """
        # 提取元数据
        metadata = self.extract_metadata_from_item(item, source_file, version)
        
        # 提取文本内容
        text_content = self.extract_text_from_item(item)
        
        # 检查是否有有效文本
        if not text_content.get("main_text") and not text_content.get("sections") and not text_content.get("paragraphs"):
            self.logger.debug(f"文档 {index} 没有有效文本内容，跳过")
            return None
            
        # 创建文档ID
        doc_id = f"{version}_{source_file.stem}_{index}"
        
        # 创建文档
        doc = EnhancedCorpusDocument(
            id=doc_id,
            text=text_content.get("main_text", ""),
            version=version,
            discipline=metadata.get("discipline", ""),
            source_file=str(source_file),
            file_name=source_file.name,
            genre="unknown",  # 默认体裁
            year=metadata.get("year"),
            date=metadata.get("date"),
            title=text_content.get("title", ""),
            abstract=text_content.get("abstract", ""),
            keywords=metadata.get("keywords", []),
            authors=metadata.get("authors", []),
            sections=text_content.get("sections", {}),
            paragraphs=text_content.get("paragraphs", []),
            metadata=metadata,
            extraction_info={
                "method": text_content.get("extraction_method"),
                "has_sections": bool(text_content.get("sections")),
                "has_paragraphs": bool(text_content.get("paragraphs")),
                "text_fields_found": list(text_content.keys())
            }
        )
        
        return doc
        
    def read_version(self, version: str, max_documents: Optional[int] = None) -> List[EnhancedCorpusDocument]:
        """
        读取整个版本的所有文档
        
        Args:
            version: 版本号
            max_documents: 最大文档数，如果为None则读取所有
            
        Returns:
            文档列表
        """
        if version not in self._versions:
            self.logger.error(f"版本 {version} 不存在")
            return []
            
        all_documents = []
        file_paths = self.list_files(version)
        
        self.logger.info(f"开始读取版本 {version}，包含 {len(file_paths)} 个文件")
        
        for file_path in file_paths:
            if max_documents is not None and len(all_documents) >= max_documents:
                break
                
            documents = self.read_file(file_path, version)
            all_documents.extend(documents)
            
            if max_documents is not None and len(all_documents) >= max_documents:
                all_documents = all_documents[:max_documents]
                break
                
        self.logger.info(f"从版本 {version} 读取了 {len(all_documents)} 个文档")
        return all_documents
        
    def read_all_versions(self, max_documents_per_version: Optional[int] = None) -> Dict[str, List[EnhancedCorpusDocument]]:
        """
        读取所有版本的文档
        
        Args:
            max_documents_per_version: 每个版本的最大文档数
            
        Returns:
            按版本分组的文档字典
        """
        all_documents = {}
        
        for version in self._versions:
            documents = self.read_version(version, max_documents_per_version)
            all_documents[version] = documents
            
        return all_documents
        
    def get_texts(self, documents: List[EnhancedCorpusDocument], 
                  include_sections: bool = False, 
                  include_paragraphs: bool = False) -> List[str]:
        """
        从文档列表中提取文本
        
        Args:
            documents: 文档列表
            include_sections: 是否包含章节内容
            include_paragraphs: 是否包含段落内容
            
        Returns:
            文本列表
        """
        texts = []
        for doc in documents:
            if include_sections or include_paragraphs:
                text = doc.get_all_text(include_sections=include_sections, include_paragraphs=include_paragraphs)
            else:
                text = doc.text
            if text:
                texts.append(text)
        return texts
        
    def get_metadata_for_analysis(self, documents: List[EnhancedCorpusDocument]) -> Dict[str, List[Any]]:
        """
        获取用于分析的元数据
        
        Args:
            documents: 文档列表
            
        Returns:
            元数据字典
        """
        metadata = {
            "version": [],
            "discipline": [],
            "source_file": [],
            "file_name": [],
            "year": [],
            "title": [],
            "authors": [],
            "keywords": []
        }
        
        for doc in documents:
            metadata["version"].append(doc.version)
            metadata["discipline"].append(doc.discipline)
            metadata["source_file"].append(doc.source_file)
            metadata["file_name"].append(doc.file_name)
            metadata["year"].append(doc.year)
            metadata["title"].append(doc.title)
            metadata["authors"].append(doc.authors)
            metadata["keywords"].append(doc.keywords)
            
        return metadata
        
    def group_by_classification(self, documents: List[EnhancedCorpusDocument], 
                               field: ClassificationField) -> Dict[str, List[EnhancedCorpusDocument]]:
        """
        按分类字段分组文档
        
        Args:
            documents: 文档列表
            field: 分类字段
            
        Returns:
            分组后的文档字典
        """
        groups = {}
        
        for doc in documents:
            value = doc.get_classification_value(field)
            if value is None:
                value = "Unknown"
            elif isinstance(value, list):
                value = str(value)
                
            key = str(value)
            if key not in groups:
                groups[key] = []
            groups[key].append(doc)
            
        return groups
        
    def filter_documents(self, documents: List[EnhancedCorpusDocument],
                        min_text_length: int = 0,
                        max_text_length: Optional[int] = None,
                        version_filter: Optional[List[str]] = None,
                        discipline_filter: Optional[List[str]] = None) -> List[EnhancedCorpusDocument]:
        """
        过滤文档
        
        Args:
            documents: 文档列表
            min_text_length: 最小文本长度
            max_text_length: 最大文本长度
            version_filter: 版本过滤器
            discipline_filter: 学科过滤器
            
        Returns:
            过滤后的文档列表
        """
        filtered = []
        
        for doc in documents:
            # 检查文本长度
            if doc.text_length < min_text_length:
                continue
                
            if max_text_length is not None and doc.text_length > max_text_length:
                continue
                
            # 检查版本过滤器
            if version_filter and doc.version not in version_filter:
                continue
                
            # 检查学科过滤器
            if discipline_filter and doc.discipline not in discipline_filter:
                continue
                
            filtered.append(doc)
            
        return filtered
        
    def get_statistics(self, documents: List[EnhancedCorpusDocument]) -> Dict[str, Any]:
        """
        获取文档统计信息
        
        Args:
            documents: 文档列表
            
        Returns:
            统计信息字典
        """
        if not documents:
            return {}
            
        # 按版本和学科分组
        version_groups = self.group_by_classification(documents, ClassificationField.VERSION)
        discipline_groups = self.group_by_classification(documents, ClassificationField.DISCIPLINE)
        
        # 文本长度统计
        text_lengths = [doc.text_length for doc in documents]
        word_counts = [doc.word_count for doc in documents]
        
        # 元数据统计
        metadata_fields = set()
        for doc in documents:
            metadata_fields.update(doc.metadata.keys())
            
        return {
            "total_documents": len(documents),
            "versions": {v: len(docs) for v, docs in version_groups.items()},
            "disciplines": {d: len(docs) for d, docs in discipline_groups.items()},
            "text_statistics": {
                "total_characters": sum(text_lengths),
                "total_words": sum(word_counts),
                "avg_characters": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                "avg_words": sum(word_counts) / len(word_counts) if word_counts else 0,
                "min_characters": min(text_lengths) if text_lengths else 0,
                "max_characters": max(text_lengths) if text_lengths else 0,
                "min_words": min(word_counts) if word_counts else 0,
                "max_words": max(word_counts) if word_counts else 0,
            },
            "metadata_fields": list(metadata_fields),
            "unique_sources": len(set(doc.source_file for doc in documents)),
            "extraction_info": {
                "documents_with_sections": sum(1 for doc in documents if doc.sections),
                "documents_with_paragraphs": sum(1 for doc in documents if doc.paragraphs),
                "documents_with_title": sum(1 for doc in documents if doc.title),
                "documents_with_abstract": sum(1 for doc in documents if doc.abstract),
            }
        }
        
    def save_documents(self, documents: List[EnhancedCorpusDocument], output_file: str):
        """
        保存文档到文件
        
        Args:
            documents: 文档列表
            output_file: 输出文件路径
        """
        serializable = [doc.to_dict() for doc in documents]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
            
        self.logger.info(f"保存了 {len(documents)} 个文档到 {output_file}")
        
    def create_analysis_ready_data(self, documents: List[EnhancedCorpusDocument]) -> Dict[str, Any]:
        """
        创建分析就绪的数据结构
        
        Args:
            documents: 文档列表
            
        Returns:
            分析就绪的数据
        """
        texts = self.get_texts(documents)
        metadata = self.get_metadata_for_analysis(documents)
        
        return {
            "texts": texts,
            "metadata": metadata,
            "document_count": len(documents),
            "statistics": self.get_statistics(documents),
            "classification_options": {
                "by_version": self.group_by_classification(documents, ClassificationField.VERSION),
                "by_discipline": self.group_by_classification(documents, ClassificationField.DISCIPLINE),
                "by_source_file": self.group_by_classification(documents, ClassificationField.SOURCE_FILE),
            }
        }


def test_enhanced_corpus_reader():
    """测试增强版语料库读取器"""
    print("测试增强版语料库读取器...")
    
    # 创建读取器
    reader = EnhancedCorpusReader("corpus", TextExtractionStrategy.SMART_EXTRACTION)
    
    # 列出版本
    versions = reader.list_versions()
    print(f"检测到的版本: {versions}")
    
    if not versions:
        print("没有找到语料库版本")
        return False
        
    # 测试读取第一个版本
    test_version = versions[0]
    print(f"\n测试读取版本: {test_version}")
    
    # 列出文件
    files = reader.list_files(test_version)
    print(f"找到 {len(files)} 个文件")
    
    if not files:
        print("没有找到数据文件")
        return False
        
    # 测试读取第一个文件
    test_file = files[0]
    print(f"\n测试读取文件: {test_file.name}")
    
    documents = reader.read_file(test_file, test_version)
    print(f"读取了 {len(documents)} 个文档")
    
    if not documents:
        print("没有读取到文档")
        return False
        
    # 显示第一个文档的信息
    first_doc = documents[0]
    print(f"\n第一个文档信息:")
    print(f"  ID: {first_doc.id}")
    print(f"  版本: {first_doc.version}")
    print(f"  学科: {first_doc.discipline}")
    print(f"  文件名: {first_doc.file_name}")
    print(f"  文本长度: {first_doc.text_length} 字符")
    print(f"  词数: {first_doc.word_count}")
    print(f"  标题: {first_doc.title[:50]}..." if first_doc.title else "  标题: 无")
    print(f"  摘要: {first_doc.abstract[:50]}..." if first_doc.abstract else "  摘要: 无")
    print(f"  关键词: {first_doc.keywords[:5]}" if first_doc.keywords else "  关键词: 无")
    print(f"  章节数: {len(first_doc.sections)}")
    print(f"  段落数: {len(first_doc.paragraphs)}")
    
    # 测试分组功能
    print(f"\n测试分组功能:")
    version_groups = reader.group_by_classification(documents, ClassificationField.VERSION)
    print(f"  按版本分组: {list(version_groups.keys())}")
    
    discipline_groups = reader.group_by_classification(documents, ClassificationField.DISCIPLINE)
    print(f"  按学科分组: {list(discipline_groups.keys())}")
    
    # 测试统计功能
    print(f"\n测试统计功能:")
    stats = reader.get_statistics(documents)
    print(f"  总文档数: {stats.get('total_documents', 0)}")
    print(f"  版本分布: {stats.get('versions', {})}")
    print(f"  学科分布: {stats.get('disciplines', {})}")
    
    # 测试分析就绪数据
    print(f"\n测试分析就绪数据:")
    analysis_data = reader.create_analysis_ready_data(documents)
    print(f"  文本数: {len(analysis_data.get('texts', []))}")
    print(f"  元数据字段: {list(analysis_data.get('metadata', {}).keys())}")
    
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    if test_enhanced_corpus_reader():
        print("\n增强版语料库读取器测试通过!")
    else:
        print("\n增强版语料库读取器测试失败!")
