# 体裁与学科分析系统

## 系统概述

本系统是一个完整的体裁与学科分析工具，专门设计用于分析语料库中的文档特征。系统支持双重分类分析（学科和体裁），提供全面的文本特征分析、可视化展示和报告生成功能。

## 系统架构

### 核心模块

1. **增强版语料库读取器** (`src/enhanced_corpus_reader.py`)
   - 支持完整元数据提取
   - 智能文本提取策略
   - 嵌套结构识别
   - 学科和体裁双重分类

2. **整合版分析系统** (`src/integrated_genre_discipline_analysis.py`)
   - 主分析引擎
   - 协调所有分析模块
   - 生成可视化图表
   - 创建分析报告

3. **分析器模块** (`src/analyzers/`)
   - 词汇分析器 (`vocabulary_analyzer.py`)
   - 增强版学术词汇分析器 (`enhanced_academic_analyzer.py`)
   - 学科比较器 (`discipline_comparator.py`)

### 数据流

```
语料库文件 (JSON)
    ↓
增强版语料库读取器
    ↓
文档对象 (包含完整元数据)
    ↓
整合版分析系统
    ├── 分类分析 (学科 + 体裁)
    ├── 词汇分析 (TTR, MTLD, 词频)
    ├── 学术词汇分析
    ├── 比较分析 (学科间 + 体裁间)
    └── 可视化生成
    ↓
输出目录 (以时间命名)
    ├── 分析报告 (Markdown)
    ├── 数据文件 (JSON, CSV)
    └── 可视化图表 (PNG)
```

## 主要功能

### 1. 双重分类分析
- **学科分类**: 按文档的学科属性分组
- **体裁分类**: 按源文件（PDF/JSON）分组，反映不同体裁特征

### 2. 文本特征分析
- **基本统计**: 文档数、文本长度、词数
- **词汇丰富度**: TTR (Type-Token Ratio)
- **文本词汇多样性**: MTLD (Measure of Textual Lexical Diversity)
- **学术词汇使用**: 学术词汇比例、学科词汇比例、交叉学科词汇比例

### 3. 比较分析
- **学科间比较**: 不同学科的文本特征差异
- **体裁间比较**: 不同源文件的文本特征差异
- **显著差异识别**: 自动识别具有显著差异的指标

### 4. 可视化展示
- **分类分布图**: 学科和体裁的饼图
- **文本特征图**: 文本长度和词数分布的直方图
- **比较分析图**: 学科间和体裁间的箱线图
- **关系图**: 文本长度与词数的散点图
- **交叉分布图**: 学科与体裁的热力图

### 5. 报告生成
- **Markdown报告**: 详细的分析结果和结论
- **JSON数据**: 完整的分析结果数据
- **CSV数据**: 关键指标的表格数据

## 使用方法

### 快速开始

```bash
# 运行演示脚本
python run_genre_discipline_analysis.py

# 直接使用整合版系统
python src/integrated_genre_discipline_analysis.py
```

### 自定义分析

```python
from src.integrated_genre_discipline_analysis import IntegratedGenreDisciplineAnalysis

# 创建分析系统
analyzer = IntegratedGenreDisciplineAnalysis("corpus")

# 运行完整分析
analyzer.run_complete_analysis(max_documents=200)

# 或分步执行
results, documents = analyzer.analyze_last_version(max_documents=200)
analyzer.create_visualizations(results, documents)
analyzer.generate_reports(results, documents)
```

## 输出结构

系统会自动创建以时间戳命名的输出目录：

```
output/
└── analysis_YYYYMMDD_HHMMSS/
    ├── analysis_report.md          # Markdown分析报告
    ├── analysis_results.json       # JSON格式完整结果
    ├── analysis_results.csv        # CSV格式关键指标
    ├── discipline_distribution.png          # 学科分布饼图
    ├── source_distribution.png             # 体裁分布饼图
    ├── text_length_histogram.png           # 文本长度直方图
    ├── word_count_histogram.png            # 词数直方图
    ├── length_vs_words_scatter.png         # 长度与词数散点图
    ├── discipline_length_boxplot.png       # 学科间长度箱线图
    ├── source_length_boxplot.png           # 体裁间长度箱线图
    └── cross_distribution_heatmap.png      # 交叉分布热力图
```

## 分析结果示例

### 关键发现

1. **文档分布**: 分析不同学科和体裁的文档数量分布
2. **文本特征**: 揭示不同分类下的文本长度、词汇丰富度等特征
3. **学术词汇**: 展示学术词汇在不同分类中的使用情况
4. **比较分析**: 量化不同学科和体裁之间的差异

### 典型输出

```
分析版本: v1.0
分析文档数: 200

学科分类 (3个学科):
  Arts & Humanities: 119个文档
  Life Sciences & Biomedicine: 80个文档
  Unknown: 1个文档

体裁分类 (3个文件):
  Arts & Humanities_random_with_pdf_text: 119个文档
  Life Sciences & Biomedicine_random_with_pdf_text: 80个文档
  example: 1个文档

词汇分析:
  总词数: 124,234
  总词型: 13,236
  TTR (词汇丰富度): 0.1065
  MTLD (文本词汇多样性): 128.57

学术词汇分析:
  总体学术词汇比例: 0.99%
  学科词汇比例: 0.00%
  交叉学科词汇比例: 0.00%

比较分析:
  文本长度差异最大的体裁:
    最长: Life Sciences & Biomedicine_random_with_pdf_text (平均 3900 字符)
    最短: Arts & Humanities_random_with_pdf_text (平均 3639 字符)
    差异: 260 字符 (7.2%)
```

## 系统优势

1. **完整性**: 从数据读取到报告生成的完整流程
2. **双重分类**: 同时支持学科和体裁分析
3. **多维度分析**: 词汇、学术、比较等多个分析维度
4. **丰富可视化**: 8种不同类型的图表展示
5. **自动化**: 自动创建时间戳目录，避免文件覆盖
6. **可扩展**: 模块化设计，易于添加新的分析功能

## 依赖要求

```txt
seaborn>=0.13.0
matplotlib>=3.7.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=1.5.0
```

## 未来扩展

1. **更多分析维度**: 添加句法分析、语义分析等
2. **交互式可视化**: 支持交互式图表和仪表板
3. **批量处理**: 支持多个语料库的批量分析
4. **API接口**: 提供REST API接口
5. **配置化**: 支持配置文件自定义分析参数

## 贡献指南

欢迎提交Issue和Pull Request来改进系统功能。

## 许可证

MIT License
