# agents/business_context_agent.py
import pandas as pd
import os
from typing import Dict, Any, List, Tuple
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from config.config import OPENAI_API_KEY, MODEL_NAME, MODEL_BASE_URL

class BusinessContextAgent:
    """业务场景识别Agent - 识别Excel数据的业务场景、提供专业化分析维度、生成分析主题和大纲"""
    
    # 支持的业务场景类型
    BUSINESS_CONTEXTS = {
        "财务会计": {
            "keywords": ["收入", "支出", "利润", "成本", "资产", "负债", "现金流", "预算", "税", "发票", "账户", "财务", "会计"],
            "dimensions": ["盈利能力分析", "成本结构分析", "现金流分析", "资产负债分析", "预算执行分析", "税务优化分析", "财务风险评估"]
        },
        "市场营销": {
            "keywords": ["销售", "市场", "客户", "产品", "广告", "促销", "渠道", "转化率", "点击", "曝光", "品牌", "竞争", "定价"],
            "dimensions": ["销售渠道分析", "客户细分分析", "产品组合分析", "营销活动效果分析", "竞争对手分析", "品牌认知度分析", "定价策略分析"]
        },
        "仓储购销": {
            "keywords": ["库存", "采购", "供应商", "仓库", "物流", "订单", "配送", "周转", "入库", "出库", "SKU", "批次"],
            "dimensions": ["库存周转分析", "供应商绩效分析", "采购成本分析", "订单履行分析", "物流效率分析", "库存优化建议", "供应链风险评估"]
        },
        "人事管理": {
            "keywords": ["员工", "招聘", "绩效", "薪资", "培训", "离职", "入职", "考勤", "人力资源", "岗位", "职级", "晋升"],
            "dimensions": ["员工流动分析", "招聘效果分析", "绩效分布分析", "薪酬结构分析", "培训投入回报分析", "人才梯队分析", "员工满意度分析"]
        },
        "行政管理": {
            "keywords": ["办公", "资产", "会议", "差旅", "报销", "审批", "合同", "文档", "规章制度", "部门"],
            "dimensions": ["行政成本分析", "资产使用效率分析", "会议效率分析", "差旅费用分析", "审批流程分析", "部门协作分析", "合规性分析"]
        },
        "产品运营": {
            "keywords": ["用户", "活跃", "留存", "转化", "注册", "DAU", "MAU", "PV", "UV", "功能", "版本", "迭代", "反馈"],
            "dimensions": ["用户增长分析", "用户留存分析", "功能使用分析", "用户行为路径分析", "版本迭代效果分析", "用户反馈分析", "运营活动效果分析"]
        },
        "教育培训": {
            "keywords": ["学生", "课程", "成绩", "教师", "班级", "考试", "作业", "出勤", "学习", "教育", "培训", "证书"],
            "dimensions": ["学生成绩分析", "课程效果分析", "教师绩效分析", "学生出勤分析", "学习进度分析", "教学质量分析", "培训投入回报分析"]
        },
        "个人日常": {
            "keywords": ["消费", "饮食", "运动", "睡眠", "娱乐", "旅行", "时间", "习惯", "日记", "个人", "家庭"],
            "dimensions": ["消费习惯分析", "时间分配分析", "健康指标分析", "生活习惯分析", "个人财务分析", "目标完成度分析", "生活质量评估"]
        },
        "医疗健康": {
            "keywords": ["患者", "医生", "诊断", "治疗", "药品", "病例", "检查", "住院", "门诊", "医院", "健康", "疾病"],
            "dimensions": ["患者流量分析", "诊疗效果分析", "药品使用分析", "医疗成本分析", "疾病分布分析", "医疗质量分析", "健康趋势分析"]
        }
    }
    
    def __init__(self):
        # 初始化LLM
        self.llm = ChatOpenAI(
            temperature=0.3,
            api_key=OPENAI_API_KEY,
            model_name=MODEL_NAME,
            base_url=MODEL_BASE_URL
        )
        
        # 创建业务场景识别提示模板
        self.context_prompt = PromptTemplate(
            input_variables=["data_info", "column_samples"],
            template="""
            请分析以下Excel数据，判断它最可能属于哪个业务场景。
            
            数据基本信息：
            {data_info}
            
            列名和样本数据：
            {column_samples}
            
            请从以下业务场景中选择最匹配的一个：
            - 财务会计：与财务报表、收支、预算、资产负债等相关
            - 市场营销：与销售、市场活动、客户、产品推广等相关
            - 仓储购销：与库存、采购、供应商、物流等相关
            - 人事管理：与员工、招聘、绩效、薪资等相关
            - 行政管理：与办公资产、会议、差旅、审批等相关
            - 产品运营：与用户活跃度、留存、功能使用等相关
            - 教育培训：与学生、课程、成绩、教师等相关
            - 个人日常：与个人消费、习惯、时间安排等相关
            - 医疗健康：与患者、诊断、治疗、药品等相关
            - 其他：如果不属于以上类别，请说明可能的业务场景
            
            请给出你的判断，并简要解释理由。同时，请提供3-5个适合这类数据的专业分析维度。
            """
        )
        
        # 创建分析维度提示模板
        self.dimension_prompt = PromptTemplate(
            input_variables=["business_context", "data_info", "column_samples"],
            template="""
            你是一位专业的{business_context}数据分析师。请为以下Excel数据提供5-7个专业的分析维度。
            
            数据基本信息：
            {data_info}
            
            列名和样本数据：
            {column_samples}
            
            请提供以下内容：
            1. 每个分析维度的名称
            2. 该维度的分析目的和业务价值
            3. 建议使用的分析方法和可视化方式
            4. 可能的洞察点和决策建议
            
            请确保分析维度专业、全面，并且与{business_context}领域的业务决策高度相关。
            """
        )
        
        # 创建主题分析提示模板
        self.theme_prompt = PromptTemplate(
            input_variables=["business_context", "data_info", "column_samples", "analysis_dimensions"],
            template="""
            你是一位专业的{business_context}数据分析师。请为以下Excel数据生成3-5个分析主题。
            
            数据基本信息：
            {data_info}
            
            列名和样本数据：
            {column_samples}
            
            已识别的分析维度：
            {analysis_dimensions}
            
            分析主题是指数据分析的核心问题或目标，它应该：
            1. 紧密结合业务场景和数据特征
            2. 能够指导后续的数据分析工作
            3. 能够产生有价值的业务洞察
            4. 可以转化为具体的分析问题
            
            请为每个主题提供：
            1. 主题名称（简洁明了）
            2. 主题描述（包括分析目的和业务价值）
            3. 相关的分析维度（从已识别的维度中选择）
            4. 可能的关键问题（该主题下需要回答的2-3个核心问题）
            
            请确保主题专业、有针对性，并且与{business_context}领域的业务决策高度相关。
            """
        )
        
        # 创建大纲生成提示模板
        self.outline_prompt = PromptTemplate(
            input_variables=["business_context", "data_info", "column_samples", "analysis_dimensions", "analysis_themes"],
            template="""
            你是一位专业的{business_context}数据分析师。请为以下Excel数据生成一份详细的分析大纲。
            
            数据基本信息：
            {data_info}
            
            列名和样本数据：
            {column_samples}
            
            已识别的分析维度：
            {analysis_dimensions}
            
            已生成的分析主题：
            {analysis_themes}
            
            请生成一份结构化的分析大纲，包括：
            
            1. 数据概览
               - 数据基本特征
               - 数据质量评估
               - 关键字段说明
            
            2. 分析方法
               - 数据预处理步骤
               - 使用的分析技术
               - 可视化方法选择
            
            3. 主题分析（针对每个主题）
               - 分析目标
               - 使用的字段和方法
               - 预期的输出和可视化
               - 关键问题的分析路径
            
            4. 综合分析
               - 跨主题的关联分析
               - 多维度交叉分析
               - 特殊场景分析
            
            5. 结论与建议框架
               - 关键发现总结方式
               - 建议生成方法
               - 决策支持要点
            
            请确保大纲专业、全面、有针对性，并且能够指导后续的数据分析工作。
            """
        )
    
    def identify_business_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """识别Excel数据的业务场景"""
        # 准备数据信息
        data_info = self._get_data_info(df)
        column_samples = self._get_column_samples(df)
        
        # 基于关键词的初步判断
        keyword_context = self._keyword_based_context(df)
        
        # 使用LLM进行更准确的判断
        context_chain = LLMChain(llm=self.llm, prompt=self.context_prompt)
        try:
            llm_result = context_chain.invoke({"data_info": data_info, "column_samples": column_samples})
            llm_context = self._extract_context_from_llm_result(llm_result["text"])
        except Exception as e:
            print(f"LLM业务场景识别失败: {str(e)}")
            llm_context = {"context": keyword_context["context"], "confidence": 0.6, "explanation": "基于关键词匹配"}
        
        # 合并结果，优先使用LLM判断
        final_context = llm_context["context"]
        
        # 获取该业务场景的专业分析维度
        analysis_dimensions = self._get_analysis_dimensions(df, final_context)
        
        return {
            "context": final_context,
            "confidence": llm_context.get("confidence", 0.8),
            "explanation": llm_context.get("explanation", ""),
            "keyword_match": keyword_context,
            "analysis_dimensions": analysis_dimensions
        }
    
    def _get_data_info(self, df: pd.DataFrame) -> str:
        """获取数据基本信息"""
        info = f"行数: {df.shape[0]}, 列数: {df.shape[1]}\n"
        info += f"列名: {', '.join(df.columns.tolist())}\n"
        info += "数据类型:\n"
        
        for col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            info += f"- {col}: {dtype}, 缺失值: {missing} ({missing_pct:.2f}%)\n"
        
        return info
    
    def _get_column_samples(self, df: pd.DataFrame) -> str:
        """获取列名和样本数据"""
        samples = ""
        sample_rows = min(5, len(df))
        
        for col in df.columns:
            samples += f"列名: {col}\n"
            samples += f"样本: {', '.join(map(str, df[col].head(sample_rows).tolist()))}\n\n"
        
        return samples
    
    def _keyword_based_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """基于关键词的业务场景初步判断"""
        # 将所有列名和样本值合并为一个文本
        text = " ".join(df.columns.tolist())
        sample_rows = min(10, len(df))
        
        for _, row in df.head(sample_rows).iterrows():
            text += " " + " ".join(map(str, row.tolist()))
        
        # 计算每个业务场景的匹配分数
        scores = {}
        for context, data in self.BUSINESS_CONTEXTS.items():
            score = 0
            for keyword in data["keywords"]:
                if keyword.lower() in text.lower():
                    score += 1
            scores[context] = score
        
        # 找出得分最高的业务场景
        max_context = max(scores.items(), key=lambda x: x[1])
        
        # 如果最高分为0，则返回未知
        if max_context[1] == 0:
            return {"context": "未知", "confidence": 0.0, "scores": scores}
        
        # 计算置信度 (0.0-1.0)
        total_keywords = sum(len(data["keywords"]) for data in self.BUSINESS_CONTEXTS.values())
        confidence = min(1.0, max_context[1] / (len(self.BUSINESS_CONTEXTS[max_context[0]]["keywords"]) * 0.7))
        
        return {
            "context": max_context[0],
            "confidence": confidence,
            "scores": scores
        }
    
    def _extract_context_from_llm_result(self, result: str) -> Dict[str, Any]:
        """从LLM结果中提取业务场景信息"""
        # 默认结果
        context_info = {"context": "未知", "confidence": 0.5, "explanation": ""}
        
        # 尝试从结果中提取业务场景
        for context in self.BUSINESS_CONTEXTS.keys():
            if context in result:
                context_info["context"] = context
                context_info["confidence"] = 0.8  # 默认置信度
                break
        
        # 提取解释部分
        explanation_lines = [line for line in result.split("\n") if line.strip() and not line.startswith("-")]
        if explanation_lines:
            context_info["explanation"] = "\n".join(explanation_lines)
        
        return context_info
    
    def _get_analysis_dimensions(self, df: pd.DataFrame, business_context: str) -> List[Dict[str, str]]:
        """获取业务场景的专业分析维度"""
        # 如果是已知业务场景，使用预定义的分析维度
        if business_context in self.BUSINESS_CONTEXTS:
            predefined_dimensions = self.BUSINESS_CONTEXTS[business_context]["dimensions"]
            dimensions = [{"name": dim, "description": ""} for dim in predefined_dimensions]
            
            # 使用LLM丰富分析维度的描述
            try:
                enriched_dimensions = self._enrich_dimensions_with_llm(df, business_context, dimensions)
                return enriched_dimensions
            except Exception as e:
                print(f"LLM丰富分析维度失败: {str(e)}")
                return dimensions
        
        # 如果是未知业务场景，使用LLM生成分析维度
        try:
            data_info = self._get_data_info(df)
            column_samples = self._get_column_samples(df)
            
            dimension_chain = LLMChain(llm=self.llm, prompt=self.dimension_prompt)
            result = dimension_chain.invoke({
                "business_context": business_context or "数据", 
                "data_info": data_info, 
                "column_samples": column_samples
            })
            
            # 解析LLM生成的分析维度
            dimensions = self._parse_llm_dimensions(result["text"])
            return dimensions
        except Exception as e:
            print(f"LLM生成分析维度失败: {str(e)}")
            return [{"name": "基础数据分析", "description": "对数据进行基本的统计分析和可视化"}]
    
    def _enrich_dimensions_with_llm(self, df: pd.DataFrame, business_context: str, 
                                   dimensions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """使用LLM丰富分析维度的描述"""
        data_info = self._get_data_info(df)
        column_samples = self._get_column_samples(df)
        
        # 构建提示
        dimension_names = "\n".join([f"- {d['name']}" for d in dimensions])
        prompt_template = PromptTemplate(
            input_variables=["business_context", "data_info", "column_samples", "dimension_names"],
            template="""
            你是一位专业的{business_context}数据分析师。请为以下分析维度提供详细的描述和分析方法。
            
            数据基本信息：
            {data_info}
            
            列名和样本数据：
            {column_samples}
            
            分析维度：
            {dimension_names}
            
            对于每个分析维度，请提供：
            1. 该维度的分析目的和业务价值
            2. 建议使用的分析方法和可视化方式
            3. 可能的洞察点和决策建议
            
            请确保描述专业、具体，并且与{business_context}领域的业务决策高度相关。
            """
        )
        
        # 创建LLM链
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        # 运行链
        result = chain.invoke({
            "business_context": business_context, 
            "data_info": data_info, 
            "column_samples": column_samples,
            "dimension_names": dimension_names
        })
        
        # 解析结果
        enriched_dimensions = self._parse_llm_dimension_descriptions(result["text"], dimensions)
        return enriched_dimensions
    
    def _parse_llm_dimensions(self, text: str) -> List[Dict[str, str]]:
        """解析LLM生成的分析维度"""
        dimensions = []
        lines = text.split("\n")
        current_dimension = None
        current_description = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是新的维度（通常以数字或破折号开头）
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "- ")) or any(line.startswith(f"{i}." ) for i in range(1, 8)):
                # 保存之前的维度
                if current_dimension and current_description:
                    dimensions.append({
                        "name": current_dimension,
                        "description": "\n".join(current_description)
                    })
                
                # 提取新维度名称
                parts = line.split(":", 1)
                if len(parts) > 1:
                    # 移除前缀数字或破折号
                    name_part = parts[0]
                    for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "- "]:
                        if name_part.startswith(prefix):
                            name_part = name_part[len(prefix):].strip()
                            break
                    
                    current_dimension = name_part
                    current_description = [parts[1].strip()]
                else:
                    current_dimension = line
                    current_description = []
            elif current_dimension:
                current_description.append(line)
        
        # 添加最后一个维度
        if current_dimension and current_description:
            dimensions.append({
                "name": current_dimension,
                "description": "\n".join(current_description)
            })
        
        return dimensions
    
    def _parse_llm_dimension_descriptions(self, text: str, dimensions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """解析LLM生成的维度描述"""
        enriched_dimensions = dimensions.copy()
        dimension_names = [d["name"] for d in dimensions]
        
        # 分割文本为段落
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue                
            # 检查段落是否包含某个维度名称
            for i, name in enumerate(dimension_names):
                if name in paragraph:
                    # 提取描述（排除维度名称所在行）
                    lines = paragraph.split("\n")
                    description_lines = []
                    for line in lines:
                        if name not in line or len(line) > len(name) + 20:  # 允许维度名称出现在较长的描述中
                            description_lines.append(line)
                    
                    if description_lines:
                        enriched_dimensions[i]["description"] = "\n".join(description_lines).strip()
                    break
        
        return enriched_dimensions
        
    def generate_analysis_themes(self, df: pd.DataFrame, business_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成数据分析主题
        
        根据业务场景和数据特征，生成3-5个分析主题，每个主题包含名称、描述、相关维度和关键问题
        """
        # 准备数据信息
        data_info = self._get_data_info(df)
        column_samples = self._get_column_samples(df)
        
        # 准备分析维度信息
        analysis_dimensions = business_context.get("analysis_dimensions", [])
        dimensions_text = "\n".join([f"- {d['name']}: {d['description']}" for d in analysis_dimensions])
        
        # 使用LLM生成分析主题
        theme_chain = LLMChain(llm=self.llm, prompt=self.theme_prompt)
        try:
            result = theme_chain.invoke({
                "business_context": business_context.get("context", "数据"),
                "data_info": data_info,
                "column_samples": column_samples,
                "analysis_dimensions": dimensions_text
            })
            
            # 解析LLM生成的分析主题
            themes = self._parse_llm_themes(result["text"])
            return themes
        except Exception as e:
            print(f"LLM生成分析主题失败: {str(e)}")
            # 返回默认主题
            return [{
                "name": "基础数据分析",
                "description": "对数据进行基本的统计分析和可视化",
                "dimensions": [d["name"] for d in analysis_dimensions[:2]] if analysis_dimensions else [],
                "key_questions": ["数据的基本分布情况是什么？", "有哪些明显的趋势和模式？"]
            }]
    
    def _parse_llm_themes(self, text: str) -> List[Dict[str, Any]]:
        """解析LLM生成的分析主题"""
        themes = []
        lines = text.split("\n")
        current_theme = None
        current_data = {}
        section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是新的主题（通常以数字或破折号开头）
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "- ")) and (":" in line or "：" in line) and not section:
                # 保存之前的主题
                if current_theme and current_data:
                    themes.append({
                        "name": current_theme,
                        **current_data
                    })
                
                # 提取新主题名称
                parts = line.split(":", 1) if ":" in line else line.split("：", 1)
                if len(parts) > 1:
                    # 移除前缀数字或破折号
                    name_part = parts[0]
                    for prefix in ["1.", "2.", "3.", "4.", "5.", "- "]:
                        if name_part.startswith(prefix):
                            name_part = name_part[len(prefix):].strip()
                            break
                    
                    current_theme = name_part
                    current_data = {"description": parts[1].strip(), "dimensions": [], "key_questions": []}
                else:
                    current_theme = line
                    current_data = {"description": "", "dimensions": [], "key_questions": []}
            
            # 检查是否是主题内的小节
            elif current_theme and (line.lower().startswith("主题描述") or 
                                  line.lower().startswith("描述") or 
                                  line.lower().startswith("相关的分析维度") or 
                                  line.lower().startswith("分析维度") or 
                                  line.lower().startswith("关键问题") or 
                                  line.lower().startswith("可能的关键问题")):
                if "描述" in line:
                    section = "description"
                elif "维度" in line:
                    section = "dimensions"
                elif "问题" in line:
                    section = "key_questions"
            
            # 处理小节内容
            elif current_theme and section and line.startswith("-"):
                content = line[1:].strip()
                if section == "description":
                    current_data["description"] += " " + content
                elif section == "dimensions":
                    current_data["dimensions"].append(content)
                elif section == "key_questions":
                    current_data["key_questions"].append(content)
            
            # 处理其他内容
            elif current_theme and section:
                if section == "description":
                    current_data["description"] += " " + line
                elif line.startswith(("1.", "2.", "3.", "4.", "5.")) and section == "key_questions":
                    current_data["key_questions"].append(line[2:].strip())
        
        # 添加最后一个主题
        if current_theme and current_data:
            themes.append({
                "name": current_theme,
                **current_data
            })
        
        return themes
    
    def generate_analysis_outline(self, df: pd.DataFrame, business_context: Dict[str, Any], 
                                analysis_themes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成数据分析大纲
        
        根据业务场景、分析维度和主题，生成结构化的分析大纲
        """
        # 准备数据信息
        data_info = self._get_data_info(df)
        column_samples = self._get_column_samples(df)
        
        # 准备分析维度信息
        analysis_dimensions = business_context.get("analysis_dimensions", [])
        dimensions_text = "\n".join([f"- {d['name']}: {d['description']}" for d in analysis_dimensions])
        
        # 准备分析主题信息
        themes_text = ""
        for i, theme in enumerate(analysis_themes):
            themes_text += f"{i+1}. {theme['name']}\n"
            themes_text += f"   描述: {theme['description']}\n"
            themes_text += f"   相关维度: {', '.join(theme.get('dimensions', []))}\n"
            themes_text += f"   关键问题: {', '.join(theme.get('key_questions', []))}\n\n"
        
        # 使用LLM生成分析大纲
        outline_chain = LLMChain(llm=self.llm, prompt=self.outline_prompt)
        try:
            result = outline_chain.invoke({
                "business_context": business_context.get("context", "数据"),
                "data_info": data_info,
                "column_samples": column_samples,
                "analysis_dimensions": dimensions_text,
                "analysis_themes": themes_text
            })
            
            # 解析LLM生成的分析大纲
            outline = self._parse_llm_outline(result["text"])
            return {
                "outline": outline,
                "raw_text": result["text"],
                "business_context": business_context.get("context", "未知"),
                "analysis_themes": [theme["name"] for theme in analysis_themes]
            }
        except Exception as e:
            print(f"LLM生成分析大纲失败: {str(e)}")
            # 返回默认大纲
            return {
                "outline": {
                    "数据概览": ["数据基本特征", "数据质量评估", "关键字段说明"],
                    "分析方法": ["数据预处理步骤", "使用的分析技术", "可视化方法选择"],
                    "主题分析": [f"主题: {theme['name']}" for theme in analysis_themes],
                    "综合分析": ["跨主题的关联分析", "多维度交叉分析"],
                    "结论与建议": ["关键发现总结", "建议生成"]
                },
                "raw_text": "",
                "business_context": business_context.get("context", "未知"),
                "analysis_themes": [theme["name"] for theme in analysis_themes]
            }
    
    def _parse_llm_outline(self, text: str) -> Dict[str, List[str]]:
        """解析LLM生成的分析大纲"""
        outline = {}
        lines = text.split("\n")
        current_section = None
        current_items = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是大纲的主要部分（通常以数字开头）
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.")) and "." in line[:3]:
                # 保存之前的部分
                if current_section and current_items:
                    outline[current_section] = current_items
                
                # 提取新部分名称
                parts = line.split(".", 1)
                if len(parts) > 1:
                    current_section = parts[1].strip()
                    current_items = []
            
            # 检查是否是子项（通常以破折号或字母开头）
            elif line.startswith("-") and current_section:
                item = line[1:].strip()
                if item:
                    current_items.append(item)
            elif line.startswith(("a.", "b.", "c.", "d.", "e.", "f.")) and current_section:
                item = line[2:].strip()
                if item:
                    current_items.append(item)
            elif line.startswith(("A.", "B.", "C.", "D.", "E.", "F.")) and current_section:
                item = line[2:].strip()
                if item:
                    current_items.append(item)
        
        # 添加最后一个部分
        if current_section and current_items:
            outline[current_section] = current_items
        
        return outline

    def generate_report_title(self, business_context, df):
        # Implement the logic to generate report title based on business context and dataframe
        title = f"{business_context['context']}数据分析报告"
        subtitle = "基于专业分析维度的深度洞察"
        return {"title": title, "subtitle": subtitle}