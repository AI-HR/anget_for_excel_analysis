# config/config.py
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 模型配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")

# LangSmith配置
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "excel_analysis_project")

# 应用配置
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 输出目录配置
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据目录配置
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 报告格式配置
REPORT_TEMPLATE = {
    "sections": [
        {
            "id": "cover_and_abstract",
            "title": "报告封面与摘要",
            "subsections": [
                {"id": "title", "title": "标题", "description": "体现核心分析主题和分析维度"}
            ]
        },
        {
            "id": "introduction_and_background",
            "title": "引言与背景",
            "word_count": "200-300",
            "subsections": [
                {"id": "analysis_background", "title": "分析背景", "description": "业务需求来源和行业趋势"},
                {"id": "analysis_goals", "title": "分析目标", "description": "3-5个核心问题和成功指标"}
            ]
        },
        {
            "id": "data_and_methods",
            "title": "数据与方法",
            "subsections": [
                {"id": "data_overview", "title": "数据概况", "description": "字段说明、数据质量和时间跨度"},
                {"id": "analysis_methods", "title": "分析方法", "description": "使用的分析方法及原因"}
            ]
        },
        {
            "id": "analysis_results",
            "title": "分析结果呈现",
            "subsections": [
                {"id": "kpi_dashboard", "title": "核心指标仪表盘", "description": "5-7个KPI看板"},
                {"id": "thematic_analysis", "title": "专题分析模块", "description": "1-3个图表/模块+解读"},
                {"id": "key_findings", "title": "关键发现标注", "description": "异常数据点和特殊事件说明"}
            ]
        },
        {
            "id": "conclusions_and_recommendations",
            "title": "结论与建议",
            "word_count": "300-400",
            "subsections": [
                {"id": "core_conclusions", "title": "核心结论", "description": "验证假设结果和战略性发现"},
                {"id": "feasible_recommendations", "title": "可行性建议", "description": "短期行动、长期规划和创新方向"}
            ]
        }
    ],
    "format_guidelines": {
        "title_font": {"name": "宋体", "bold": True, "size": 2},  # 二号字
        "level1_font": {"name": "宋体", "bold": True, "size": 3.5},  # 小三号字
        "level1_line_spacing": 1.5,
        "level2_font": {"name": "宋体", "bold": True, "size": 3},  # 三号字
        "level2_line_spacing": 1.2,
        "body_font": {"name": "宋体", "bold": False, "size": 3},  # 三号字
        "body_line_spacing": 1.0,
        "body_first_line_indent": 2  # 首行缩进2个字符
    }
}
