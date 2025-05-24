# Excel智能分析报告生成系统

## 项目介绍

本项目是一个基于LLM的Excel智能分析报告生成系统，能够自动识别Excel文件的业务场景（如财务会计、市场营销、人事管理等），并根据不同业务场景提供专业化的多维度分析结果。

## 项目介绍

这是一个基于LLM（大型语言模型）的Excel数据智能分析系统，能够自动对上传的Excel文件进行数据清洗、分析、可视化，并生成专业的分析报告。系统采用Agent架构，由多个专业Agent协同工作，完成端到端的数据分析流程。

## 系统架构

系统由以下几个主要组件构成：

1. **主控Agent (MasterAgent)**: 协调所有专业Agent的工作，管理整个分析流程
2. **业务场景识别Agent (BusinessContextAgent)**: 识别Excel数据所属的业务场景，提供专业分析维度
3. **数据清洗Agent (DataCleaningAgent)**: 负责数据预处理，包括处理缺失值、异常值等
4. **数据分析Agent (AnalysisAgent)**: 执行统计分析、相关性分析等数据分析任务
5. **数据可视化Agent (VisualizationAgent)**: 创建各种可视化图表，展示数据特征和分析结果
6. **报告生成Agent (ReportAgent)**: 整合业务场景、分析结果和可视化图表，生成专业的分析报告

## 功能特点

- **业务场景自动识别**：能够根据Excel数据特征自动判断所属业务领域（财务会计、市场营销、仓储购销、人事管理等）
- **专业化分析维度**：针对不同业务场景提供专业的分析维度和指标体系
- **自动数据清洗**：处理缺失值、异常值、数据类型转换等
- **智能数据分析**：执行描述性统计、相关性分析、分布分析、时间序列分析等
- **多样化可视化**：生成柱状图、折线图、散点图、箱线图、热力图、饼图等
- **专业报告生成**：自动生成包含业务场景分析、专业维度解读和针对性建议的Word报告
- **Web界面**：提供友好的Web界面，支持文件上传和结果展示

## 安装步骤

1. 克隆项目代码

```bash
git clone <项目仓库URL>
cd data_analysis
```

2. 安装依赖包

```bash
pip install -r requirements.txt
```

3. 配置环境变量

创建`.env`文件，设置以下环境变量：

```
OPENAI_API_KEY=你的OpenAI API密钥
MODEL_NAME=你要使用的模型名称
MODEL_BASE_URL=模型API的基础URL
OUTPUT_DIR=输出目录路径
```

## 使用方法

### 命令行方式

```bash
python main.py --file 你的Excel文件路径
```

### Web界面方式

1. 启动Web服务

```bash
python app.py
```

2. 在浏览器中访问 http://localhost:8000
3. 上传Excel文件并等待分析完成
4. 查看分析结果和下载报告

## 支持的业务场景

系统能够自动识别以下业务场景，并提供相应的专业分析维度：

### 财务会计
- 盈利能力分析
- 成本结构分析
- 现金流分析
- 资产负债分析
- 预算执行分析
- 税务优化分析
- 财务风险评估

### 市场营销
- 销售渠道分析
- 客户细分分析
- 产品组合分析
- 营销活动效果分析
- 竞争对手分析
- 品牌认知度分析
- 定价策略分析

### 仓储购销
- 库存周转分析
- 供应商绩效分析
- 采购成本分析
- 订单履行分析
- 物流效率分析
- 库存优化建议
- 供应链风险评估

### 人事管理
- 员工流动分析
- 招聘效果分析
- 绩效分布分析
- 薪酬结构分析
- 培训投入回报分析
- 人才梯队分析
- 员工满意度分析

### 其他业务场景
- 行政管理
- 产品运营
- 教育培训
- 个人日常
- 医疗健康

## 项目结构

```
data_analysis/
├── .env                  # 环境变量配置文件
├── .env.example          # 环境变量示例文件
├── agents/               # Agent实现
│   ├── analysis_agent.py       # 数据分析Agent
│   ├── business_context_agent.py  # 业务场景识别Agent
│   ├── data_cleaning_agent.py  # 数据清洗Agent
│   ├── master_agent.py         # 主控Agent
│   ├── report_agent.py         # 报告生成Agent
│   └── visualization_agent.py  # 数据可视化Agent
├── config/               # 配置文件
│   └── config.py         # 配置参数
├── models/               # 数据模型
├── routes/               # API路由
│   └── api_routes.py     # API接口定义
├── services/             # 服务层
├── utils/                # 工具函数
├── templates/            # HTML模板
│   └── index.html        # 首页模板
├── static/               # 静态资源
├── app.py                # Web应用入口
├── main.py               # 命令行入口
└── requirements.txt      # 依赖包列表
```

## 技术栈

- **Python**: 核心编程语言
- **LangChain**: LLM应用框架
- **OpenAI API**: 大型语言模型服务
- **Pandas & NumPy**: 数据处理
- **Matplotlib & Seaborn**: 数据可视化
- **FastAPI**: Web框架
- **python-docx**: Word文档生成

## 注意事项

- 系统需要联网以访问OpenAI API
- 处理大型Excel文件可能需要较长时间
- 确保有足够的磁盘空间存储生成的报告和图表

## 许可证

[MIT License](LICENSE)