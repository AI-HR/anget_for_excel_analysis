# agents/analysis_agent.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config.config import OPENAI_API_KEY, MODEL_NAME, MODEL_BASE_URL

class AnalysisAgent:
    """基于LLM的智能数据分析Agent"""
    
    def __init__(self):
        # 初始化LLM
        self.llm = ChatOpenAI(
            temperature=0.2,
            api_key=OPENAI_API_KEY,
            model_name=MODEL_NAME,
            base_url=MODEL_BASE_URL
        )
        
        # 当前数据框
        self.current_df = None
        
        # 定义分析工具
        self.tools = [
            Tool(
                name="描述性统计分析",
                func=self._descriptive_analysis,
                description="执行描述性统计分析，计算均值、中位数、标准差等统计量"
            ),
            Tool(
                name="相关性分析",
                func=self._correlation_analysis,
                description="分析变量之间的相关关系，计算相关系数"
            ),
            Tool(
                name="分布分析",
                func=self._distribution_analysis,
                description="分析数据分布特征，包括正态性检验、分位数等"
            ),
            Tool(
                name="方差分析",
                func=self._anova_analysis,
                description="比较不同组之间的差异，执行ANOVA分析"
            ),
            Tool(
                name="时间序列分析",
                func=self._time_series_analysis,
                description="分析时间序列数据的趋势、季节性和周期性"
            ),
            Tool(
                name="聚类分析",
                func=self._clustering_analysis,
                description="对数据进行聚类，识别相似组"
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
        
    def analyze_data(self, df: pd.DataFrame, columns: List[str] = None, business_context: Optional[Dict[str, Any]] = None, 
                   analysis_themes: Optional[List[Dict[str, Any]]] = None, 
                   analysis_outline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """使用Agent智能分析数据，根据业务场景、分析主题和大纲进行更有针对性的分析"""
        # 设置当前数据框，供工具函数使用
        self.current_df = df
        
        # 准备业务场景信息
        context_info = ""
        if business_context and "context" in business_context:
            context_info = f"业务场景: {business_context['context']}\n"
            if "explanation" in business_context and business_context["explanation"]:
                context_info += f"场景解释: {business_context['explanation']}\n"
            if "analysis_dimensions" in business_context and business_context["analysis_dimensions"]:
                context_info += "专业分析维度:\n"
                for dimension in business_context['analysis_dimensions']:
                    context_info += f"- {dimension['name']}\n"
                    if "description" in dimension and dimension['description']:
                        context_info += f"  {dimension['description']}\n"
        
        # 准备分析主题信息
        themes_info = ""
        if analysis_themes:
            themes_info = "分析主题:\n"
            for theme in analysis_themes:
                themes_info += f"- {theme['name']}: {theme['description']}\n"
                if "key_questions" in theme and theme["key_questions"]:
                    themes_info += f"  关键问题: {', '.join(theme['key_questions'])}\n"
        
        # 准备分析大纲信息
        outline_info = ""
        if analysis_outline and "outline" in analysis_outline:
            outline_info = "分析大纲结构:\n"
            for section, items in analysis_outline["outline"].items():
                outline_info += f"- {section}: {', '.join(items)}\n"
        
        # 准备列信息
        columns_info = "\n列名及数据类型:\n"
        for col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            columns_info += f"- {col}: {dtype}, 缺失值: {missing} ({missing_pct:.2f}%)\n"
        
        # 构建提示
        prompt = f"""
        请作为一位专业的数据分析师，对以下数据集进行全面分析。
        
        {context_info}
        
        {themes_info}
        
        {outline_info}
        
        {columns_info}
        
        请根据业务场景、分析主题和大纲，执行以下分析任务：
        1. 描述性统计分析：计算每个数值列的基本统计量（均值、中位数、标准差、最小值、最大值等）
        2. 分布分析：分析每个重要特征的分布情况
        3. 相关性分析：计算数值特征之间的相关性，识别高相关性特征对
        4. 分组分析：根据分类变量进行分组，分析各组的差异
        5. 时间序列分析：如果有时间特征，分析时间趋势和模式
        6. 主题分析：针对每个分析主题，进行专项分析
        
        请特别注意：
        - 分析应紧密结合业务场景和分析主题
        - 优先分析大纲中提到的关键部分
        - 确保分析结果具有业务价值和可解释性
        - 识别数据中的关键模式、趋势和异常
        
        请提供详细的分析计划，包括每个步骤的具体操作和预期输出。
        """
        
        # 使用LLM生成分析计划
        try:
            analysis_plan = self.llm.invoke(prompt)
            print("生成数据分析计划...")
            # 执行分析计划
            analysis_results = self._execute_analysis_plan(df, analysis_plan.content, business_context, analysis_themes, analysis_outline)
            return analysis_results
        except Exception as e:
            print(f"生成分析计划失败: {str(e)}")
            # 如果LLM调用失败，执行基本的分析
            return self._basic_analysis(df, None, business_context, analysis_themes, analysis_outline)
    
    def _execute_analysis_plan(self, df: pd.DataFrame, analysis_plan: str, 
                             business_context: Optional[Dict[str, Any]] = None,
                             analysis_themes: Optional[List[Dict[str, Any]]] = None,
                             analysis_outline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行数据分析计划，根据业务场景、分析主题和大纲进行更有针对性的分析"""
        print("执行数据分析计划...")
        analysis_results = {}
        
        # 解析分析计划并执行
        try:
            # 准备业务场景信息
            business_info = ""
            if business_context:
                business_info = f"""
                业务场景信息：
                - 场景: {business_context.get('context', '未指定')}
                - 解释: {business_context.get('explanation', '未提供')}
                """
                if 'analysis_dimensions' in business_context and business_context['analysis_dimensions']:
                    business_info += "- 专业分析维度:\n"
                    for dim in business_context['analysis_dimensions']:
                        business_info += f"  * {dim.get('name', '')}: {dim.get('description', '')}\n"
            
            # 准备分析主题信息
            themes_info = ""
            if analysis_themes:
                themes_info = "分析主题信息:\n"
                for theme in analysis_themes:
                    themes_info += f"- {theme.get('name', '')}: {theme.get('description', '')}\n"
                    if 'key_questions' in theme and theme['key_questions']:
                        themes_info += f"  关键问题: {', '.join(theme['key_questions'])}\n"
            
            # 准备分析大纲信息
            outline_info = ""
            if analysis_outline and 'outline' in analysis_outline:
                outline_info = "分析大纲结构:\n"
                for section, items in analysis_outline["outline"].items():
                    outline_info += f"- {section}: {', '.join(items)}\n"
            
            # 构建执行分析计划的提示，包含业务场景和分析主题信息
            prompt = f"""
            请根据以下数据分析计划和业务场景信息，生成可执行的Python代码来分析数据。
            
            分析计划：
            {analysis_plan}
            
            {business_info}
            
            {themes_info}
            
            {outline_info}
            
            请生成完整的Python代码，使用pandas和numpy库分析数据。代码应该接收一个名为'df'的DataFrame作为输入，
            并返回一个包含所有分析结果的字典。字典的结构应该清晰，便于后续处理。
            
            分析结果应包括：
            1. 描述性统计结果
            2. 相关性分析结果
            3. 分组分析结果
            4. 时间序列分析结果（如果适用）
            5. 针对每个分析主题的专项分析结果
            
            请确保分析结果与业务场景紧密结合，并针对分析主题提供深入见解。
            如果有分析大纲，请按照大纲结构组织分析结果。
            
            只返回可执行的Python代码，不要包含任何解释或注释。
            """
            
            # 使用LLM生成分析代码
            response = self.llm.invoke(prompt)
            analysis_code = response.content
            
            # 提取代码部分（如果LLM返回了Markdown格式的代码块）
            if "```python" in analysis_code and "```" in analysis_code:
                analysis_code = analysis_code.split("```python")[1].split("```")[0].strip()
            elif "```" in analysis_code:
                analysis_code = analysis_code.split("```")[1].split("```")[0].strip()
            
            # 创建本地变量以供执行
            local_vars = {
                "df": df, 
                "pd": pd, 
                "np": np, 
                "business_context": business_context,
                "analysis_themes": analysis_themes,
                "analysis_outline": analysis_outline
            }
            
            # 执行分析代码
            exec(analysis_code, globals(), local_vars)
            
            # 获取分析结果
            if "analysis_results" in local_vars:
                analysis_results = local_vars["analysis_results"]
            elif "results" in local_vars:
                analysis_results = local_vars["results"]
            
            # 添加元数据
            analysis_results["metadata"] = {
                "generated_at": pd.Timestamp.now().isoformat(),
                "data_shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            # 添加业务场景信息
            if business_context:
                analysis_results["business_context"] = business_context
            
            # 添加分析主题信息
            if analysis_themes:
                analysis_results["analysis_themes"] = analysis_themes
            
            # 添加分析大纲信息
            if analysis_outline:
                analysis_results["analysis_outline"] = analysis_outline
            
            print(f"数据分析完成，生成了{len(analysis_results)}个分析结果")
            return analysis_results
            
        except Exception as e:
            print(f"执行分析计划失败: {str(e)}")
            print("执行基本分析...")
            return self._basic_analysis(df, business_context, analysis_themes, analysis_outline)
    
    def _basic_analysis(self, df: pd.DataFrame, columns: List[str] = None, business_context: Optional[Dict[str, Any]] = None, 
                      analysis_themes: Optional[List[Dict[str, Any]]] = None, 
                      analysis_outline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行基本的数据分析，考虑业务场景、分析主题和大纲"""
        print("执行基本数据分析...")
        analysis_results = {}
        
        # 添加元数据
        analysis_results["metadata"] = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "data_shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "analysis_type": "basic",  # 标记为基本分析
        }
        
        # 1. 描述性统计
        try:
            descriptive = {}
            for col in df.select_dtypes(include=['number']).columns:
                descriptive[col] = {
                    "count": df[col].count(),
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "25%": df[col].quantile(0.25),
                    "50%": df[col].median(),
                    "75%": df[col].quantile(0.75),
                    "max": df[col].max()
                }
            analysis_results["descriptive"] = descriptive
        except Exception as e:
            print(f"描述性统计分析失败: {str(e)}")
        
        # 2. 相关性分析
        try:
            # 计算数值列的相关性
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                corr_matrix = numeric_df.corr()
                
                # 提取高相关性特征对（相关系数绝对值大于0.7）
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr_pairs.append({
                                "feature1": corr_matrix.columns[i],
                                "feature2": corr_matrix.columns[j],
                                "correlation": corr_matrix.iloc[i, j]
                            })
                
                analysis_results["correlation"] = {
                    "matrix": corr_matrix.to_dict(),
                    "high_correlation_pairs": high_corr_pairs
                }
        except Exception as e:
            print(f"相关性分析失败: {str(e)}")
        
        # 3. 分类变量分析
        try:
            categorical_analysis = {}
            for col in df.select_dtypes(include=['object', 'category']).columns:
                if df[col].nunique() < 20:  # 只分析取值较少的分类变量
                    value_counts = df[col].value_counts()
                    categorical_analysis[col] = {
                        "value_counts": value_counts.to_dict(),
                        "unique_count": df[col].nunique()
                    }
            analysis_results["categorical"] = categorical_analysis
        except Exception as e:
            print(f"分类变量分析失败: {str(e)}")
        
        # 4. 时间序列分析（如果有日期列）
        try:
            # 尝试识别日期列，包括已经是日期类型的和可以转换为日期类型的
            date_columns = list(df.select_dtypes(include=['datetime64']).columns)
            
            # 尝试将字符串列转换为日期
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # 检查是否可以转换为日期
                    if pd.to_datetime(df[col], errors='coerce').notna().mean() > 0.8:  # 如果80%以上的值可以转换为日期
                        date_columns.append(col)
                except:
                    pass
            
            if date_columns:
                time_analysis = {}
                for date_col in date_columns:
                    # 确保列是日期类型
                    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                        date_series = pd.to_datetime(df[date_col], errors='coerce')
                    else:
                        date_series = df[date_col]
                    
                    # 跳过包含太多NaT的列
                    if date_series.isna().mean() > 0.2:  # 如果超过20%的值是NaT
                        continue
                    
                    # 提取时间特征
                    time_features = {}
                    
                    # 按年分析
                    if date_series.dt.year.nunique() > 1:
                        yearly_counts = df.groupby(date_series.dt.year).size().to_dict()
                        time_features["yearly"] = yearly_counts
                    
                    # 按月分析
                    if date_series.dt.month.nunique() > 1:
                        monthly_counts = df.groupby(date_series.dt.month).size().to_dict()
                        time_features["monthly"] = monthly_counts
                        
                        # 按月分析数值列的平均值
                        for num_col in df.select_dtypes(include=['number']).columns:
                            monthly_avgs = df.groupby(date_series.dt.month)[num_col].mean().to_dict()
                            time_features[f"{num_col}_monthly_avg"] = monthly_avgs
                    
                    # 按日分析
                    if date_series.dt.day.nunique() > 1:
                        daily_counts = df.groupby(date_series.dt.day).size().to_dict()
                        time_features["daily"] = daily_counts
                    
                    time_analysis[date_col] = time_features
                
                if time_analysis:  # 只有在有分析结果时才添加
                    analysis_results["time_series"] = time_analysis
        except Exception as e:
            print(f"时间序列分析失败: {str(e)}")
        
        # 5. 分组分析 - 根据分类变量对数值变量进行分组分析
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if not categorical_cols.empty and not numeric_cols.empty:
                group_analysis = {}
                
                for cat_col in categorical_cols:
                    # 只分析取值较少的分类变量
                    if df[cat_col].nunique() < 10 and df[cat_col].nunique() > 1:
                        group_stats = {}
                        for num_col in numeric_cols:
                            # 计算每个组的基本统计量
                            group_means = df.groupby(cat_col)[num_col].mean().to_dict()
                            group_medians = df.groupby(cat_col)[num_col].median().to_dict()
                            group_stds = df.groupby(cat_col)[num_col].std().to_dict()
                            
                            group_stats[num_col] = {
                                "mean": group_means,
                                "median": group_medians,
                                "std": group_stds
                            }
                        
                        group_analysis[cat_col] = group_stats
                
                if group_analysis:  # 只有在有分析结果时才添加
                    analysis_results["group_analysis"] = group_analysis
        except Exception as e:
            print(f"分组分析失败: {str(e)}")
        
        # 6. 业务场景相关分析
        if business_context:
            try:
                business_analysis = {
                    "context": business_context.get("context", ""),
                    "key_metrics": {}
                }
                
                # 根据业务场景识别关键指标
                context_type = business_context.get("context", "").lower()
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                # 财务分析场景
                if any(term in context_type for term in ["财务", "金融", "会计", "资金", "预算"]):
                    # 计算基本财务指标
                    for col in numeric_cols:
                        col_lower = col.lower()
                        if any(term in col_lower for term in ["收入", "revenue", "income", "销售", "sale"]):
                            business_analysis["key_metrics"][f"{col}_总和"] = df[col].sum()
                            business_analysis["key_metrics"][f"{col}_平均值"] = df[col].mean()
                        elif any(term in col_lower for term in ["利润", "profit", "margin", "收益"]):
                            business_analysis["key_metrics"][f"{col}_总和"] = df[col].sum()
                            business_analysis["key_metrics"][f"{col}_平均值"] = df[col].mean()
                        elif any(term in col_lower for term in ["成本", "cost", "expense", "支出"]):
                            business_analysis["key_metrics"][f"{col}_总和"] = df[col].sum()
                            business_analysis["key_metrics"][f"{col}_平均值"] = df[col].mean()
                
                # 销售/营销场景
                elif any(term in context_type for term in ["销售", "营销", "市场", "客户", "消费者"]):
                    for col in numeric_cols:
                        col_lower = col.lower()
                        if any(term in col_lower for term in ["销量", "sales", "订单", "order"]):
                            business_analysis["key_metrics"][f"{col}_总和"] = df[col].sum()
                            business_analysis["key_metrics"][f"{col}_平均值"] = df[col].mean()
                        elif any(term in col_lower for term in ["价格", "price", "单价", "rate"]):
                            business_analysis["key_metrics"][f"{col}_平均值"] = df[col].mean()
                            business_analysis["key_metrics"][f"{col}_中位数"] = df[col].median()
                
                # 人力资源场景
                elif any(term in context_type for term in ["人力", "人事", "员工", "招聘", "绩效"]):
                    for col in numeric_cols:
                        col_lower = col.lower()
                        if any(term in col_lower for term in ["薪资", "salary", "工资", "报酬"]):
                            business_analysis["key_metrics"][f"{col}_平均值"] = df[col].mean()
                            business_analysis["key_metrics"][f"{col}_中位数"] = df[col].median()
                            business_analysis["key_metrics"][f"{col}_标准差"] = df[col].std()
                        elif any(term in col_lower for term in ["年龄", "age", "工龄", "经验"]):
                            business_analysis["key_metrics"][f"{col}_平均值"] = df[col].mean()
                            business_analysis["key_metrics"][f"{col}_中位数"] = df[col].median()
                
                # 添加业务分析结果
                if business_analysis["key_metrics"]:  # 只有在有分析结果时才添加
                    analysis_results["business_analysis"] = business_analysis
            except Exception as e:
                print(f"业务场景分析失败: {str(e)}")
        
        # 7. 主题分析
        if analysis_themes:
            try:
                themes_analysis = {}
                
                for theme in analysis_themes:
                    theme_name = theme.get("name", "")
                    theme_desc = theme.get("description", "")
                    key_questions = theme.get("key_questions", [])
                    
                    theme_result = {
                        "description": theme_desc,
                        "key_findings": {}
                    }
                    
                    # 根据主题关键词选择相关列进行分析
                    theme_keywords = theme_name.lower().split() + theme_desc.lower().split()
                    theme_keywords = [kw for kw in theme_keywords if len(kw) > 2]  # 过滤掉太短的词
                    
                    # 找出与主题相关的列
                    related_cols = []
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(kw in col_lower for kw in theme_keywords):
                            related_cols.append(col)
                    
                    # 如果找到相关列，进行分析
                    if related_cols:
                        for col in related_cols:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                theme_result["key_findings"][f"{col}_统计"] = {
                                    "mean": df[col].mean(),
                                    "median": df[col].median(),
                                    "std": df[col].std(),
                                    "min": df[col].min(),
                                    "max": df[col].max()
                                }
                            elif pd.api.types.is_object_dtype(df[col]) and df[col].nunique() < 20:
                                theme_result["key_findings"][f"{col}_分布"] = df[col].value_counts().to_dict()
                    
                    # 如果没有找到相关列，使用所有数值列
                    else:
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        for col in numeric_cols:
                            theme_result["key_findings"][f"{col}_基本统计"] = {
                                "mean": df[col].mean(),
                                "median": df[col].median(),
                                "std": df[col].std()
                            }
                    
                    themes_analysis[theme_name] = theme_result
                
                if themes_analysis:  # 只有在有分析结果时才添加
                    analysis_results["themes_analysis"] = themes_analysis
            except Exception as e:
                print(f"主题分析失败: {str(e)}")
        
        # 8. 根据分析大纲组织结果
        if analysis_outline and "outline" in analysis_outline:
            try:
                outline_results = {}
                
                for section, items in analysis_outline["outline"].items():
                    section_results = {}
                    
                    for item in items:
                        item_lower = item.lower()
                        item_results = {}
                        
                        # 查找与大纲项相关的列
                        related_cols = []
                        for col in df.columns:
                            col_lower = col.lower()
                            if item_lower in col_lower or any(word in col_lower for word in item_lower.split()):
                                related_cols.append(col)
                        
                        # 对相关列进行分析
                        for col in related_cols:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                item_results[col] = {
                                    "mean": df[col].mean(),
                                    "median": df[col].median(),
                                    "std": df[col].std(),
                                    "min": df[col].min(),
                                    "max": df[col].max()
                                }
                            elif pd.api.types.is_object_dtype(df[col]) and df[col].nunique() < 20:
                                item_results[col] = {
                                    "value_counts": df[col].value_counts().to_dict(),
                                    "unique_count": df[col].nunique()
                                }
                        
                        # 如果没有找到相关列，添加一个占位符
                        if not item_results:
                            item_results["note"] = "未找到与该项直接相关的数据列"
                        
                        section_results[item] = item_results
                    
                    outline_results[section] = section_results
                
                analysis_results["outline_analysis"] = outline_results
            except Exception as e:
                print(f"大纲分析失败: {str(e)}")
        
        # 添加业务场景、分析主题和大纲信息
        if business_context:
            analysis_results["business_context"] = business_context
        if analysis_themes:
            analysis_results["analysis_themes"] = analysis_themes
        if analysis_outline:
            analysis_results["analysis_outline"] = analysis_outline
        
        print(f"基本数据分析完成，生成了{len(analysis_results)}个分析结果")
        return analysis_results
        
        if columns is None:
            columns = df.columns.tolist()
            
        # 数据概览
        data_info = self._get_data_overview(df[columns])
        
        # 构建Agent提示
        prompt = f"""
        我需要对一个数据集进行全面分析。数据集信息如下：
        
        {data_info}
        
        请根据数据特征，选择最合适的分析方法，并执行以下任务：
        1. 确定哪些分析方法适合这些数据
        2. 执行相应的分析
        3. 识别数据中的关键模式、趋势和关系
        4. 提供分析结果的简要解释
        
        请给出详细的分析计划和每一步的理由。
        """
        
        # 运行Agent获取分析计划
        try:
            analysis_plan = self.agent.invoke(prompt)
            # 执行分析计划
            analysis_results = self._execute_analysis_plan(df, columns, analysis_plan)
            return analysis_results
        except Exception as e:
            print(f"分析数据时出错: {str(e)}")
            # 如果Agent调用失败，执行基本的数据分析
            return self._basic_analysis(df, columns)
    
    def _get_data_overview(self, df: pd.DataFrame) -> str:
        """获取数据概览信息"""
        overview = f"行数: {df.shape[0]}, 列数: {df.shape[1]}\n"
        overview += f"列名: {', '.join(df.columns.tolist())}\n"
        overview += "数据类型:\n"
        
        for col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            overview += f"- {col}: {dtype}, 缺失值: {missing} ({missing_pct:.2f}%)\n"
            
            # 对数值列添加基本统计信息
            if pd.api.types.is_numeric_dtype(df[col]):
                overview += f"  范围: {df[col].min()} - {df[col].max()}, 均值: {df[col].mean():.2f}, 标准差: {df[col].std():.2f}\n"
            
            # 对分类列添加唯一值信息
            elif pd.api.types.is_object_dtype(df[col]):
                unique_count = df[col].nunique()
                overview += f"  唯一值数量: {unique_count}\n"
                if unique_count <= 10:  # 如果唯一值较少，显示所有值
                    unique_values = df[col].unique()
                    overview += f"  唯一值: {', '.join(str(v) for v in unique_values if pd.notna(v))}\n"
                
        return overview
    
    def _execute_analysis_plan(self, df: pd.DataFrame, analysis_plan: str, business_context: Optional[Dict[str, Any]] = None, 
                      analysis_themes: Optional[List[Dict[str, Any]]] = None, 
                      analysis_outline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行数据分析计划"""
        results = {}
        
        # 获取所有列
        columns = df.columns.tolist()
        
        # 执行描述性统计分析
        results["descriptive"] = self._descriptive_analysis_impl(df, columns)
        
        # 执行相关性分析（仅对数值列）
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            results["correlation"] = self._correlation_analysis_impl(df, numeric_cols)
        
        # 执行分布分析（仅对数值列）
        if numeric_cols:
            results["distribution"] = self._distribution_analysis_impl(df, numeric_cols)
        
        # 检查是否有分类列和数值列，如果有则执行方差分析
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols and numeric_cols:
            results["anova"] = self._anova_analysis_impl(df, categorical_cols[0], numeric_cols[0])
        
        # 检查是否有日期列，如果有则执行时间序列分析
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        if date_cols and numeric_cols:
            results["time_series"] = self._time_series_analysis_impl(df, date_cols[0], numeric_cols[0])
        
        # 如果有足够的数值列，执行聚类分析
        if len(numeric_cols) >= 2:
            results["clustering"] = self._clustering_analysis_impl(df, numeric_cols)
        
        # 添加业务场景、分析主题和大纲信息
        if business_context:
            results["business_context"] = business_context
        if analysis_themes:
            results["analysis_themes"] = analysis_themes
        if analysis_outline:
            results["analysis_outline"] = analysis_outline
        
        return results
    
    # 工具函数实现
    def _descriptive_analysis(self, query: str) -> str:
        """描述性统计分析工具函数"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        result = self._descriptive_analysis_impl(self.current_df)
        return f"已完成描述性统计分析，计算了均值、中位数、标准差等统计量。\n{str(result)[:500]}"
    
    def _correlation_analysis(self, query: str) -> str:
        """相关性分析工具函数"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        numeric_cols = self.current_df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            return "错误：需要至少两个数值列才能进行相关性分析"
        
        result = self._correlation_analysis_impl(self.current_df, numeric_cols)
        return f"已完成相关性分析，计算了变量之间的相关系数。\n{str(result)[:500]}"
    
    def _distribution_analysis(self, query: str) -> str:
        """分布分析工具函数"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        numeric_cols = self.current_df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            return "错误：需要至少一个数值列才能进行分布分析"
        
        result = self._distribution_analysis_impl(self.current_df, numeric_cols)
        return f"已完成分布分析，分析了数据分布特征。\n{str(result)[:500]}"
    
    def _anova_analysis(self, query: str) -> str:
        """方差分析工具函数"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        categorical_cols = self.current_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = self.current_df.select_dtypes(include=['number']).columns.tolist()
        
        if not categorical_cols or not numeric_cols:
            return "错误：需要至少一个分类列和一个数值列才能进行方差分析"
        
        result = self._anova_analysis_impl(self.current_df, categorical_cols[0], numeric_cols[0])
        return f"已完成方差分析，比较了不同组之间的差异。\n{str(result)[:500]}"
    
    def _time_series_analysis(self, query: str) -> str:
        """时间序列分析工具函数"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        date_cols = self.current_df.select_dtypes(include=['datetime']).columns.tolist()
        numeric_cols = self.current_df.select_dtypes(include=['number']).columns.tolist()
        
        if not date_cols or not numeric_cols:
            return "错误：需要至少一个日期列和一个数值列才能进行时间序列分析"
        
        result = self._time_series_analysis_impl(self.current_df, date_cols[0], numeric_cols[0])
        return f"已完成时间序列分析，分析了趋势、季节性和周期性。\n{str(result)[:500]}"
    
    def _clustering_analysis(self, query: str) -> str:
        """聚类分析工具函数"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        numeric_cols = self.current_df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            return "错误：需要至少两个数值列才能进行聚类分析"
        
        result = self._clustering_analysis_impl(self.current_df, numeric_cols)
        return f"已完成聚类分析，识别了相似组。\n{str(result)[:500]}"
    
    # 实际实现函数
    def _descriptive_analysis_impl(self, df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """实际执行描述性统计分析的实现"""
        if columns is None:
            columns = df.columns.tolist()
            
        # 基本描述性统计
        desc_stats = df[columns].describe().to_dict()
        
        # 添加额外的统计量
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                desc_stats.setdefault(col, {})
                desc_stats[col]['skewness'] = df[col].skew()
                desc_stats[col]['kurtosis'] = df[col].kurtosis()
                desc_stats[col]['missing_count'] = df[col].isna().sum()
                desc_stats[col]['missing_percent'] = (df[col].isna().sum() / len(df)) * 100
        
        return desc_stats
    
    def _correlation_analysis_impl(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """实际执行相关性分析的实现"""
        # 计算相关系数矩阵
        corr_matrix = df[columns].corr().to_dict()
        
        # 找出高相关性的特征对
        high_corr_pairs = []
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:  # 避免重复
                    corr = df[col1].corr(df[col2])
                    if abs(corr) > 0.7:  # 高相关性阈值
                        high_corr_pairs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': corr
                        })
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs
        }
    
    def _distribution_analysis_impl(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """实际执行分布分析的实现"""
        from scipy import stats
        
        distribution_results = {}
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # 移除缺失值
                data = df[col].dropna()
                
                # 跳过空列
                if len(data) == 0:
                    continue
                    
                # 正态性检验
                try:
                    shapiro_test = stats.shapiro(data)
                    normality = {
                        'shapiro_statistic': shapiro_test[0],
                        'shapiro_pvalue': shapiro_test[1],
                        'is_normal': shapiro_test[1] > 0.05
                    }
                except:
                    normality = {'error': '无法执行正态性检验'}
                
                # 分位数
                quantiles = {
                    'q1': data.quantile(0.25),
                    'q2': data.quantile(0.5),
                    'q3': data.quantile(0.75),
                    'iqr': data.quantile(0.75) - data.quantile(0.25)
                }
                
                distribution_results[col] = {
                    'normality': normality,
                    'quantiles': quantiles,
                    'skewness': data.skew(),
                    'kurtosis': data.kurtosis()
                }
        
        return distribution_results
    
    def _anova_analysis_impl(self, df: pd.DataFrame, category_col: str, value_col: str) -> Dict[str, Any]:
        """实际执行方差分析的实现"""
        from scipy import stats
        
        # 确保分类列和数值列存在
        if category_col not in df.columns or value_col not in df.columns:
            return {'error': '指定的列不存在'}
        
        # 确保数值列是数值类型
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return {'error': '指定的值列不是数值类型'}
        
        # 按分类列分组
        groups = df.groupby(category_col)[value_col].apply(list).to_dict()
        
        # 过滤掉空组
        groups = {k: v for k, v in groups.items() if len(v) > 0}
        
        # 如果组数小于2，无法进行ANOVA
        if len(groups) < 2:
            return {'error': '分组数量不足，无法进行方差分析'}
        
        # 执行单因素方差分析
        try:
            f_statistic, p_value = stats.f_oneway(*groups.values())
            
            # 计算每组的基本统计量
            group_stats = {}
            for group, values in groups.items():
                group_stats[group] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            return {
                'f_statistic': f_statistic,
                'p_value': p_value,
                'significant_difference': p_value < 0.05,
                'group_stats': group_stats
            }
        except Exception as e:
            return {'error': f'执行方差分析时出错: {str(e)}'}
    
    def _time_series_analysis_impl(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """实际执行时间序列分析的实现"""
        # 确保日期列和数值列存在
        if date_col not in df.columns or value_col not in df.columns:
            return {'error': '指定的列不存在'}
        
        # 确保日期列是日期类型，数值列是数值类型
        if not pd.api.types.is_datetime64_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except:
                return {'error': '无法将指定的列转换为日期类型'}
        
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return {'error': '指定的值列不是数值类型'}
        
        # 按日期排序
        df_sorted = df.sort_values(by=date_col)
        
        # 设置日期为索引
        df_ts = df_sorted.set_index(date_col)[[value_col]]
        
        # 基本时间序列统计
        ts_stats = {
            'start_date': df_ts.index.min().strftime('%Y-%m-%d'),
            'end_date': df_ts.index.max().strftime('%Y-%m-%d'),
            'duration_days': (df_ts.index.max() - df_ts.index.min()).days,
            'count': len(df_ts),
            'mean': df_ts[value_col].mean(),
            'std': df_ts[value_col].std(),
            'min': df_ts[value_col].min(),
            'max': df_ts[value_col].max()
        }
        
        # 简单的趋势分析
        try:
            # 创建一个数值索引
            df_ts['index'] = range(len(df_ts))
            
            # 线性回归
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_ts['index'], df_ts[value_col])
            
            trend_analysis = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'significant_trend': p_value < 0.05
            }
        except Exception as e:
            trend_analysis = {'error': f'执行趋势分析时出错: {str(e)}'}
        
        return {
            'time_series_stats': ts_stats,
            'trend_analysis': trend_analysis
        }
    
    def _clustering_analysis_impl(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """实际执行聚类分析的实现"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # 确保所有列都是数值类型
        numeric_df = df[columns].select_dtypes(include=['number'])
        if numeric_df.shape[1] < 2:
            return {'error': '需要至少两个数值列才能进行聚类分析'}
        
        # 处理缺失值：移除包含NaN的行
        numeric_df = numeric_df.dropna()
        
        # 确保数据不为空再进行标准化
        if numeric_df.empty:
            return {'error': '没有可用的数值数据进行聚类分析。'}

        # 标准化数据
        scaler = StandardScaler()
        try:
            scaled_data = scaler.fit_transform(numeric_df)
        except ValueError as e:
            return {'error': f'数据标准化时出错: {str(e)}'}
        
        # 确定最佳聚类数（简化版，使用肘部法则）
        max_clusters = min(10, len(numeric_df) // 10)  # 最多10个聚类或数据量的1/10
        if max_clusters < 2:
            max_clusters = 2
            
        inertia = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)
        
        # 简单的肘部法则实现
        optimal_k = 2  # 默认值
        for i in range(1, len(inertia) - 1):
            prev_diff = inertia[i-1] - inertia[i]
            next_diff = inertia[i] - inertia[i+1]
            if prev_diff > 2 * next_diff:
                optimal_k = i + 1
                break
        
        # 使用最佳聚类数进行聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # 计算每个聚类的基本统计信息
        df_with_clusters = numeric_df.copy()
        df_with_clusters['cluster'] = clusters
        
        cluster_stats = {}
        for cluster_id in range(optimal_k):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            cluster_stats[f'cluster_{cluster_id}'] = {
                'count': len(cluster_data),
                'percent': (len(cluster_data) / len(df_with_clusters)) * 100
            }
            
            # 计算每个特征的均值
            for col in numeric_df.columns:
                cluster_stats[f'cluster_{cluster_id}'][f'{col}_mean'] = cluster_data[col].mean()
        
        return {
            'optimal_clusters': optimal_k,
            'inertia': inertia,
            'cluster_stats': cluster_stats,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_labels': clusters.tolist()
        }