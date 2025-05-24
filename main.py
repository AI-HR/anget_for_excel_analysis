# main.py
import os
import pandas as pd
import argparse
from typing import Dict, Any
from agents.master_agent import MasterAgent
from agents.data_cleaning_agent import DataCleaningAgent
from agents.analysis_agent import AnalysisAgent
from agents.visualization_agent import VisualizationAgent
from agents.report_agent import ReportAgent
from agents.business_context_agent import BusinessContextAgent
from config.config import OUTPUT_DIR

class DataAnalysisSystem:
    """Excel智能分析报告生成系统"""
    
    def __init__(self):
        # 初始化各个Agent
        self.data_cleaning_agent = DataCleaningAgent()
        self.analysis_agent = AnalysisAgent()
        self.visualization_agent = VisualizationAgent()
        self.business_context_agent = BusinessContextAgent()
        self.report_agent = ReportAgent()
        
        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def analyze_excel(self, file_path: str) -> Dict[str, Any]:
        """分析Excel文件并生成报告"""
        print(f"开始分析Excel文件: {file_path}")
        
        # 读取Excel文件
        try:
            df = pd.read_excel(file_path)
            print(f"成功读取Excel文件，共{df.shape[0]}行，{df.shape[1]}列")
        except Exception as e:
            print(f"读取Excel文件时出错: {str(e)}")
            return {"error": f"读取Excel文件时出错: {str(e)}"}
        
        # 识别业务场景
        try:
            print("识别数据业务场景...")
            business_context = self.business_context_agent.identify_business_context(df)
            print(f"识别到业务场景: {business_context['context']}，置信度: {business_context['confidence']:.2f}")
            print(f"业务分析维度: {', '.join([d['name'] for d in business_context['analysis_dimensions']])}")
        except Exception as e:
            print(f"业务场景识别时出错: {str(e)}")
            business_context = {"context": "未知", "confidence": 0.0, "analysis_dimensions": []}
        
        # 数据清洗
        try:
            print("开始数据清洗...")
            cleaned_df = self.data_cleaning_agent.clean_data(df)
            print(f"数据清洗完成，处理后数据集: {cleaned_df.shape[0]}行，{cleaned_df.shape[1]}列")
        except Exception as e:
            print(f"数据清洗时出错: {str(e)}")
            return {"error": f"数据清洗时出错: {str(e)}"}
        
        # 数据分析
        try:
            print("开始数据分析...")
            analysis_results = self.analysis_agent.analyze_data(cleaned_df)
            # 添加业务场景信息到分析结果中
            analysis_results["business_context"] = business_context
            print("数据分析完成")
        except Exception as e:
            print(f"数据分析时出错: {str(e)}")
            return {"error": f"数据分析时出错: {str(e)}"}
        
        # 数据可视化
        try:
            print("开始数据可视化...")
            visualization_results = self.visualization_agent.visualize_data(cleaned_df, analysis_results)
            print("数据可视化完成")
        except Exception as e:
            print(f"数据可视化时出错: {str(e)}")
            return {"error": f"数据可视化时出错: {str(e)}"}
        
        # 生成报告
        try:
            print("开始生成报告...")
            report_results = self.report_agent.generate_report(cleaned_df, analysis_results, visualization_results)
            print("报告生成完成")
        except Exception as e:
            print(f"生成报告时出错: {str(e)}")
            return {"error": f"生成报告时出错: {str(e)}"}
        
        # 返回结果
        return {
            "cleaned_data": cleaned_df,
            "business_context": business_context,
            "analysis_results": analysis_results,
            "visualization_results": visualization_results,
            "report_results": report_results
        }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Excel智能分析报告生成系统")
    parser.add_argument("--file", "-f", type=str, required=True, help="Excel文件路径")
    args = parser.parse_args()
    
    # 初始化系统
    system = DataAnalysisSystem()
    
    # 分析Excel文件
    results = system.analyze_excel(args.file)
    
    # 输出结果
    if "error" in results:
        print(f"错误: {results['error']}")
    else:
        report_path = results["report_results"].get("word_report_path")
        if report_path:
            print(f"\n分析报告已生成: {report_path}")
        else:
            print("\n分析完成，但未能生成报告文件")

if __name__ == "__main__":
    main()