# agents/master_agent.py
import pandas as pd
from typing import Dict, Any, List, Optional
import os
import uuid
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from agents.data_cleaning_agent import DataCleaningAgent
from agents.analysis_agent import AnalysisAgent
from agents.visualization_agent import VisualizationAgent
from agents.report_agent import ReportAgent
from agents.business_context_agent import BusinessContextAgent
from config.config import OPENAI_API_KEY, MODEL_NAME, MODEL_BASE_URL

class MasterAgent:
    """主控Agent - 协调所有专业Agent的工作"""
    
    def __init__(self):
        # 初始化各个专业Agent
        self.data_cleaning_agent = DataCleaningAgent()
        self.analysis_agent = AnalysisAgent()
        self.visualization_agent = VisualizationAgent()
        self.business_context_agent = BusinessContextAgent()
        self.report_agent = ReportAgent()
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            temperature=0.3,
            api_key=OPENAI_API_KEY,
            model_name=MODEL_NAME,
            base_url=MODEL_BASE_URL,
            model_kwargs={
                "max_tokens": 4000
            }
        )
        
        # 定义工具
        self.tools = [
            Tool(
                name="业务场景识别",
                func=self._identify_business_context,
                description="识别数据的业务场景，如财务会计、市场营销、人事管理等"
            ),
            Tool(
                name="数据清洗",
                func=self._clean_data,
                description="清洗和预处理数据，处理缺失值、异常值等"
            ),
            Tool(
                name="数据分析",
                func=self._analyze_data,
                description="分析数据，执行统计分析、相关性分析等"
            ),
            Tool(
                name="数据可视化",
                func=self._visualize_data,
                description="创建数据可视化图表，如柱状图、折线图、散点图等"
            ),
            Tool(
                name="生成报告",
                func=self._generate_report,
                description="生成数据分析报告，包括Word文档"
            ),
            Tool(
                name="提供建议",
                func=self._provide_recommendations,
                description="根据数据分析结果提供业务建议和洞察"
            )
        ]
        
        # 初始化Agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
    def process_data_file(self, file_path: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        """处理数据文件，执行端到端的数据分析流程"""
        # 加载数据
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return {"error": "不支持的文件格式"}
        except Exception as e:
            return {"error": f"加载数据失败: {str(e)}"}

        # 剖析用户需求
        user_analysis_request = user_query or ""
        print(f"用户需求: {user_analysis_request}")

        # 构建提示
        preview_rows = min(5, len(df))
        prompt = f"""
            我需要对一个数据文件进行分析。文件: {file_path}
            数据预览:
            {df.iloc[:preview_rows].to_string()}
            基本统计: 行数:{len(df)}, 列数:{df.shape[1]}
            用户查询: {user_analysis_request}
            任务:
            1. 识别业务场景
            2. 清洗数据
            3. 分析数据
            4. 创建可视化
            5. 生成报告
            6. 提供建议
            请提供分析计划。
        """

        # 运行Agent获取处理计划
        processing_plan = self.agent.invoke({"input": prompt})

        # 执行处理计划
        result = self._execute_processing_plan(df, processing_plan, user_analysis_request)
        return result
    
    def _execute_processing_plan(self, df: pd.DataFrame, processing_plan: Dict, user_query: Optional[str] = None) -> Dict[str, Any]:
        """执行数据处理计划"""
        results = {}
        
        # 1. 识别业务场景
        try:
            print("识别数据业务场景...")
            business_context = self.business_context_agent.identify_business_context(df)
            print(f"识别到业务场景: {business_context['context']}，置信度: {business_context['confidence']:.2f}")
            print(f"业务分析维度: {', '.join([d['name'] for d in business_context['analysis_dimensions']])}")
            results["business_context"] = business_context
        except Exception as e:
            print(f"业务场景识别失败: {str(e)}")
            results["business_context"] = {"error": str(e)}
            business_context = {"context": "未知", "confidence": 0.0, "analysis_dimensions": []}
        
        # 1.1 生成分析主题
        try:
            print("生成分析主题...")
            analysis_themes = self.business_context_agent.generate_analysis_themes(df, business_context)
            print(f"生成分析主题: {', '.join([theme['name'] for theme in analysis_themes])}")
            results["analysis_themes"] = analysis_themes
        except Exception as e:
            print(f"生成分析主题失败: {str(e)}")
            results["analysis_themes"] = {"error": str(e)}
            analysis_themes = [{
                "name": "基础数据分析",
                "description": "对数据进行基本的统计分析和可视化",
                "dimensions": [],
                "key_questions": ["数据的基本分布情况是什么？", "有哪些明显的趋势和模式？"]
            }]
        
        # 1.2 生成分析大纲
        try:
            print("生成分析大纲...")
            analysis_outline = self.business_context_agent.generate_analysis_outline(df, business_context, analysis_themes)
            print(f"生成分析大纲，包含{len(analysis_outline['outline'])}个主要部分")
            results["analysis_outline"] = analysis_outline
        except Exception as e:
            print(f"生成分析大纲失败: {str(e)}")
            results["analysis_outline"] = {"error": str(e)}
            analysis_outline = {
                "outline": {
                    "数据概览": ["数据基本特征", "数据质量评估"],
                    "分析方法": ["数据预处理步骤", "使用的分析技术"],
                    "主题分析": [f"主题: {theme['name']}" for theme in analysis_themes],
                    "结论与建议": ["关键发现总结", "建议生成"]
                }
            }
        
        # 2. 数据清洗
        try:
            print("开始数据清洗...")
            # 将分析大纲传递给数据清洗Agent，以便根据大纲智能选择相关字段进行处理
            cleaned_df = self.data_cleaning_agent.clean_data(df, analysis_outline=analysis_outline)
            results["cleaned_data"] = {
                "rows": len(cleaned_df),
                "columns": cleaned_df.shape[1],
                "preview": cleaned_df.head(3).to_dict()
            }
        except Exception as e:
            print(f"数据清洗失败: {str(e)}")
            results["cleaned_data"] = {"error": str(e)}
            cleaned_df = df  # 如果清洗失败，使用原始数据继续
        
        # 3. 数据分析
        try:
            print("开始数据分析...")
            # 将业务场景、分析主题和大纲传递给分析Agent，以便进行更有针对性的分析
            analysis_results = self.analysis_agent.analyze_data(
                cleaned_df, 
                business_context=business_context,
                analysis_themes=analysis_themes,
                analysis_outline=analysis_outline
            )
            # 添加业务场景信息到分析结果中
            analysis_results["business_context"] = business_context
            analysis_results["analysis_themes"] = analysis_themes
            analysis_results["analysis_outline"] = analysis_outline
            results["analysis"] = analysis_results
        except Exception as e:
            print(f"数据分析失败: {str(e)}")
            results["analysis"] = {"error": str(e)}
            analysis_results = {}
        
        # 4. 数据可视化
        try:
            print("开始数据可视化...")
            # 将业务场景、分析主题和大纲传递给可视化Agent，以便生成更有针对性的可视化
            visualization_results = self.visualization_agent.create_visualizations(
                cleaned_df, 
                analysis_results,
                business_context=business_context,
                analysis_themes=analysis_themes,
                analysis_outline=analysis_outline
            )
            results["visualizations"] = visualization_results
        except Exception as e:
            print(f"数据可视化失败: {str(e)}")
            results["visualizations"] = {"error": str(e)}
            visualization_results = []
        
        # 5. 生成报告
        try:
            print("开始生成报告...")
            context_name = business_context.get("context", "未知")
            report_title = f"数据分析报告 - {context_name}业务场景 - {user_query or '自动分析'}"
            # 将业务场景、分析主题和大纲传递给报告Agent，以便生成更专业、更有针对性的报告
            report_results = self.report_agent.generate_report(
                cleaned_df, 
                analysis_results, 
                visualization_results,
                report_title,
                business_context=business_context,
                analysis_themes=analysis_themes,
                analysis_outline=analysis_outline
            )
            results["report"] = report_results
        except Exception as e:
            print(f"生成报告失败: {str(e)}")
            results["report"] = {"error": str(e)}
        
        return results
    
    # 工具函数实现
    def _identify_business_context(self, query: str) -> str:
        """识别业务场景工具函数"""
        return "业务场景识别完成，已确定数据所属业务领域和专业分析维度。"
    
    def _clean_data(self, query: str) -> str:
        """清洗数据工具函数"""
        return "数据清洗完成，已处理缺失值和异常值。"
    
    def _analyze_data(self, query: str) -> str:
        """分析数据工具函数"""
        return "数据分析完成，已执行描述性统计和相关性分析。"
    
    def _visualize_data(self, query: str) -> str:
        """可视化数据工具函数"""
        return "数据可视化完成，已创建关键指标的图表。"
    
    def _generate_report(self, query: str) -> str:
        """生成报告工具函数"""
        return "报告生成完成，包含分析结果和可视化图表。"
    
    def _provide_recommendations(self, query: str) -> str:
        """提供建议工具函数"""
        return "根据分析结果，建议关注以下几个关键指标..."