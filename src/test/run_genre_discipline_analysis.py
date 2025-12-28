#!/usr/bin/env python3
"""
run_genre_discipline_analysis.py - 运行体裁与学科分析系统

这是一个简单的演示脚本，展示如何使用整合版体裁与学科分析系统。
"""

import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, 'src')

def main():
    """主函数"""
    print("体裁与学科分析系统演示")
    print("=" * 80)
    
    try:
        # 导入整合版分析系统
        from integrated_genre_discipline_analysis import IntegratedGenreDisciplineAnalysis
        
        print("1. 创建分析系统...")
        analyzer = IntegratedGenreDisciplineAnalysis("corpus")
        
        print("2. 运行完整分析...")
        print("   这将执行以下步骤:")
        print("   - 读取语料库文档")
        print("   - 分析学科和体裁分类")
        print("   - 进行词汇和学术词汇分析")
        print("   - 比较不同学科和体裁的特征")
        print("   - 生成可视化图表")
        print("   - 创建分析报告")
        
        # 运行分析
        analyzer.run_complete_analysis(max_documents=1e15)
        
        print("\n3. 分析完成!")
        print("=" * 80)
        print("生成的文件:")
        
        # 列出生成的文件
        output_dir = analyzer.output_dir
        if output_dir.exists():
            print(f"输出目录: {output_dir}")
            print("\n生成的文件列表:")
            
            # 列出所有文件
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    size_kb = file_path.stat().st_size / 1024
                    print(f"  - {file_path.name} ({size_kb:.1f} KB)")
                    
            # 显示报告内容摘要
            report_file = output_dir / "analysis_report.md"
            if report_file.exists():
                print(f"\n报告摘要:")
                with open(report_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:20]  # 读取前20行
                    for line in lines:
                        if line.strip():
                            print(f"  {line.strip()}")
                            
        print("\n4. 系统功能总结:")
        print("   - 增强版语料库读取器: 支持完整元数据提取")
        print("   - 双重分类分析: 同时分析学科和体裁")
        print("   - 词汇分析: TTR、MTLD、词频等指标")
        print("   - 学术词汇分析: 学术词汇使用情况")
        print("   - 比较分析: 学科间和体裁间差异")
        print("   - 可视化: 8种不同类型的图表")
        print("   - 报告生成: JSON、CSV、Markdown格式")
        print("   - 时间戳输出: 自动创建时间戳目录")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所有依赖:")
        print("  pip install seaborn matplotlib numpy scipy")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
