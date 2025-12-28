# 体裁与学科分析系统（清理版）

## 系统概述

这是一个经过清理和优化的体裁与学科分析系统，专门用于分析语料库中的文档特征。系统支持双重分类分析（学科和体裁），提供全面的文本特征分析、可视化展示和报告生成功能。

## 清理说明

已删除以下无用文件：
- 旧的测试文件 (`test_*.py`, `demo_*.py`)
- 重复的系统文件 (`main_analyzer.py`, `corpus_reader.py`, `visualization.py`, `StatisticCorpus.py`, `analyzer_base.py`, `analyzer_manager.py`)
- 旧的配置文件和数据文件 (`analysis_config.example.json`, `corpus_statistics.json`, `filtered_output.json`)
- 旧的系统文档 (`DISPLINE_ANALYSIS_SYSTEM.md`, `ANALYSIS_SYSTEM_README.md`)
- 旧的输出文件（保留时间戳目录）
- Python缓存文件 (`__pycache__` 目录)

## 核心文件结构

```
GenreShift/
├── run_genre_discipline_analysis.py      # 演示脚本
├── GENRE_DISCIPLINE_ANALYSIS_SYSTEM.md   # 系统文档
├── README.md                             # 项目说明
├── LICENSE                               # 许可证
├── .gitignore                           # Git忽略文件
├── .clinerules/                         # 项目规则
├── corpus/                              # 语料库数据
├── output/                              # 输出目录（自动创建时间戳子目录）
└── src/                                 # 源代码
    ├── enhanced_corpus_reader.py         # 增强版语料库读取器
    ├── enhanced_visualization.py         # 增强版可视化模块
    ├── integrated_genre_discipline_analysis.py  # 整合版分析系统
    ├── analyzers/                        # 分析器模块
    │   ├── vocabulary_analyzer.py        # 词汇分析器
    │   ├── academic_vocabulary_analyzer.py      # 学术词汇分析器
    │   ├── enhanced_academic_analyzer.py        # 增强版学术词汇分析器
    │   └── discipline_comparator.py      # 学科比较器
    └── wordlists/                        # 词表工具
        └── wordlist_downloader.py        # 词表下载器
```

## 快速开始

```bash
# 运行演示脚本
python run_genre_discipline_analysis.py

# 或直接使用整合版系统
python src/integrated_genre_discipline_analysis.py
```

## 系统功能

1. **双重分类分析**
   - 学科分类：按文档的学科属性分组
   - 体裁分类：按源文件（PDF/JSON）分组

2. **多维度文本分析**
   - 基本统计：文档数、文本长度、词数
   - 词汇丰富度：TTR (Type-Token Ratio)
   - 文本词汇多样性：MTLD (Measure of Textual Lexical Diversity)
   - 学术词汇使用：学术词汇比例、学科词汇比例、交叉学科词汇比例

3. **比较分析**
   - 学科间比较：不同学科的文本特征差异
   - 体裁间比较：不同源文件的文本特征差异
   - 显著差异识别：自动识别具有显著差异的指标

4. **可视化展示**（8种图表）
   - 分类分布图：学科和体裁的饼图
   - 文本特征图：文本长度和词数分布的直方图
   - 比较分析图：学科间和体裁间的箱线图
   - 关系图：文本长度与词数的散点图
   - 交叉分布图：学科与体裁的热力图

5. **报告生成**
   - Markdown报告：详细的分析结果和结论
   - JSON数据：完整的分析结果数据
   - CSV数据：关键指标的表格数据

6. **时间戳输出**
   - 所有输出自动保存到以时间命名的文件夹中
   - 避免文件覆盖，便于版本管理

## 输出示例

系统运行后会生成类似以下结构的输出：

```
output/
└── analysis_20251227_222206/
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

## 依赖要求

```txt
seaborn>=0.13.0
matplotlib>=3.7.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=1.5.0
```

安装依赖：
```bash
pip install seaborn matplotlib numpy scipy pandas
```

## 系统优势

1. **简洁高效**：删除无用文件，保留核心功能
2. **双重分类**：同时支持学科和体裁分析
3. **多维度分析**：词汇、学术、比较等多个分析维度
4. **丰富可视化**：8种不同类型的图表展示
5. **自动化**：自动创建时间戳目录，避免文件覆盖
6. **模块化**：清晰的模块结构，易于维护和扩展

## 使用示例

```python
from src.integrated_genre_discipline_analysis import IntegratedGenreDisciplineAnalysis

# 创建分析系统
analyzer = IntegratedGenreDisciplineAnalysis("corpus")

# 运行完整分析（限制200个文档以加快速度）
analyzer.run_complete_analysis(max_documents=200)

# 输出目录
print(f"分析结果保存在: {analyzer.output_dir}")
```

## 清理验证

系统已成功清理，保留了所有核心功能：
- ✅ 增强版语料库读取器
- ✅ 整合版分析系统
- ✅ 所有分析器模块
- ✅ 可视化功能
- ✅ 报告生成功能
- ✅ 时间戳输出功能
- ✅ 演示脚本
- ✅ 系统文档

系统现在更加简洁、高效，易于使用和维护。
