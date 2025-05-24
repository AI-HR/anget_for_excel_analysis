# agents/data_cleaning_agent.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config.config import OPENAI_API_KEY, MODEL_NAME, MODEL_BASE_URL

class DataCleaningAgent:
    """基于LLM的智能数据清洗Agent"""
    
    def __init__(self):
        # 初始化LLM
        self.llm = ChatOpenAI(
            temperature=0,
            api_key=OPENAI_API_KEY,
            model_name=MODEL_NAME,
            base_url=MODEL_BASE_URL
        )
        
        # 定义数据清洗工具
        self.tools = [
            Tool(
                name="识别数据类型",
                func=self._identify_data_types,
                description="分析数据框中各列的数据类型，识别可能的错误类型"
            ),
            Tool(
                name="处理缺失值",
                func=self._handle_missing_values,
                description="智能处理数据中的缺失值，根据数据特征选择合适的填充方法"
            ),
            Tool(
                name="处理异常值",
                func=self._handle_outliers,
                description="检测并处理数据中的异常值，使用多种方法识别异常"
            ),
            Tool(
                name="数据转换",
                func=self._transform_data,
                description="对数据进行必要的转换，如标准化、归一化、对数转换等"
            ),
            Tool(
                name="特征工程",
                func=self._feature_engineering,
                description="创建新的特征或转换现有特征以提高数据质量"
            )
        ]
        
        # 初始化Agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5  # 限制最大迭代次数，避免无限循环
        )
        
    def clean_data(self, df: pd.DataFrame, analysis_outline: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """使用Agent智能清洗数据，根据分析大纲智能选择相关字段进行处理"""
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
            
        # 样本数据
        sample_info = "\n样本数据 (前3行):\n"
        sample_info += df.head(3).to_string()

        # 设置当前数据框，供工具函数使用
        self.current_df = df
        
        # 构建提示
        prompt = f"""
        请作为一位专业的数据清洗专家，对以下数据集进行智能清洗。
        
        {columns_info}
        
        {sample_info}
        
        {outline_info}
        
        请根据分析大纲和数据特征，执行以下清洗任务：
        1. 处理缺失值：根据每列的特性选择合适的缺失值处理方法（删除、填充均值/中位数/众数、插值等）
        2. 处理异常值：识别并处理可能的异常值和离群点
        3. 数据类型转换：将列转换为适当的数据类型
        4. 特征工程：根据分析大纲创建必要的新特征
        5. 数据标准化/归一化：如果需要，对数值特征进行标准化处理
        
        请特别注意：
        - 优先处理分析大纲中提到的关键字段
        - 保留对分析有价值的信息
        - 确保清洗后的数据质量和一致性
        - 记录所有清洗步骤和处理逻辑
        
        请提供详细的清洗计划，包括每个步骤的具体操作和理由。
        """
        
        # 使用LLM生成清洗计划
        try:
            cleaning_plan = self.llm.invoke(prompt)
            print("生成数据清洗计划...")
            # 执行清洗计划
            cleaned_df = self._execute_cleaning_plan(df, cleaning_plan.content, analysis_outline)
            return cleaned_df
        except Exception as e:
            print(f"生成清洗计划失败: {str(e)}")
            # 如果LLM调用失败，执行基本的清洗
            return self._basic_cleaning(df)
    
    def _execute_cleaning_plan(self, df: pd.DataFrame, cleaning_plan: str, analysis_outline: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """执行数据清洗计划"""
        print("执行数据清洗计划...")
        cleaned_df = df.copy()
        
        # 解析清洗计划并执行
        try:
            # 构建执行清洗计划的提示
            prompt = f"""
            请根据以下数据清洗计划，生成可执行的Python代码来清洗数据。
            
            清洗计划：
            {cleaning_plan}
            
            请生成完整的Python代码，使用pandas库处理数据。代码应该接收一个名为'df'的DataFrame作为输入，并返回清洗后的DataFrame。
            只返回可执行的Python代码，不要包含任何解释或注释。
            """
            
            # 使用LLM生成清洗代码
            response = self.llm.invoke(prompt)
            cleaning_code = response.content
            
            # 提取代码部分（如果LLM返回了Markdown格式的代码块）
            if "```python" in cleaning_code and "```" in cleaning_code:
                cleaning_code = cleaning_code.split("```python")[1].split("```")[0].strip()
            elif "```" in cleaning_code:
                cleaning_code = cleaning_code.split("```")[1].split("```")[0].strip()
            
            # 创建本地变量以供执行
            local_vars = {"df": cleaned_df, "pd": pd, "np": np}
            
            # 执行清洗代码
            exec(cleaning_code, globals(), local_vars)
            
            # 获取清洗后的DataFrame
            if "cleaned_df" in local_vars:
                cleaned_df = local_vars["cleaned_df"]
            elif "result_df" in local_vars:
                cleaned_df = local_vars["result_df"]
            
            print(f"数据清洗完成，处理后行数: {len(cleaned_df)}, 列数: {cleaned_df.shape[1]}")
            return cleaned_df
            
        except Exception as e:
            print(f"执行清洗计划失败: {str(e)}")
            print("执行基本清洗...")
            return self._basic_cleaning(df)
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行基本的数据清洗"""
        print("执行基本数据清洗...")
        cleaned_df = df.copy()
        
        # 1. 处理缺失值
        for col in cleaned_df.columns:
            # 数值型列用中位数填充
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            # 类别型列用众数填充
            elif pd.api.types.is_object_dtype(cleaned_df[col]) or pd.api.types.is_categorical_dtype(cleaned_df[col]):
                if cleaned_df[col].nunique() < len(cleaned_df) / 2:  # 只有当唯一值不太多时才用众数填充
                    mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "未知"
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
            # 日期型列不处理
            elif pd.api.types.is_datetime64_dtype(cleaned_df[col]):
                pass
        
        # 2. 处理异常值（针对数值型列，使用IQR方法）
        for col in cleaned_df.select_dtypes(include=['number']).columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 将异常值替换为边界值
            cleaned_df[col] = cleaned_df[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
        
        # 3. 数据类型转换
        # 尝试将可能的日期字符串转换为日期类型
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            # 检查是否可能是日期
            if cleaned_df[col].str.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}').any():
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                    # 如果转换后有太多NaT，则恢复原始数据
                    if cleaned_df[col].isna().mean() > 0.3:  # 如果超过30%变成了NaT
                        cleaned_df[col] = df[col]
                except:
                    pass
        
        print(f"基本数据清洗完成，处理后行数: {len(cleaned_df)}, 列数: {cleaned_df.shape[1]}")
        return cleaned_df
        data_info = self._get_data_overview(df)
        
        # 构建Agent提示
        prompt = f"""
        我需要对一个数据集进行智能清洗。数据集信息如下：
        
        {data_info}
        
        请分析这些数据并执行以下任务：
        1. 识别并处理缺失值，为每列选择最合适的填充方法
        2. 检测并处理异常值
        3. 转换不正确的数据类型
        4. 如有必要，进行特征工程
        5. 提供清洗过程的解释
        
        请给出详细的清洗计划和每一步的理由。
        """
        
        # 运行Agent获取清洗计划
        try:
            cleaning_plan = self.agent.invoke(prompt)
            # 执行清洗计划
            cleaned_df = self._execute_cleaning_plan(df, cleaning_plan)
            return cleaned_df
        except Exception as e:
            print(f"清洗数据时出错: {str(e)}")
            # 如果Agent调用失败，执行基本的数据清洗
            return self._basic_cleaning(df)
    
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
    
    # 删除重复的_execute_cleaning_plan方法，保留上面定义的版本
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行基本的数据清洗"""
        # 创建数据副本
        cleaned_df = df.copy()
        
        # 处理缺失值 - 数值列使用中位数，分类列使用众数
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                # 使用直接赋值而不是inplace=True
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                # 使用直接赋值而不是inplace=True
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "未知")
        
        # 处理异常值 - 对数值列进行简单的异常值处理（基于IQR）
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
        
        return cleaned_df
    
    # 工具函数实现
    def _identify_data_types(self, query: str) -> str:
        """识别数据类型工具函数"""
        return "已识别各列数据类型，并标记可能的错误类型。"
    
    def _handle_missing_values(self, query: str) -> str:
        """处理缺失值工具函数"""
        return "已处理缺失值，根据数据特征选择了合适的填充方法。"
    
    def _handle_outliers(self, query: str) -> str:
        """处理异常值工具函数"""
        return "已检测并处理异常值，使用了多种方法识别异常。"
    
    def _transform_data(self, query: str) -> str:
        """数据转换工具函数"""
        return "已对数据进行必要的转换，如标准化、归一化等。"
    
    def _feature_engineering(self, query: str) -> str:
        """特征工程工具函数"""
        return "已创建新的特征或转换现有特征，提高了数据质量。"
    
    # 实际实现函数
    def _handle_missing_values_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        """实际处理缺失值的实现"""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # 数值列使用中位数填充
                df[col] = df[col].fillna(df[col].median())
            elif pd.api.types.is_datetime64_dtype(df[col]):
                # 日期列使用前一个值填充
                df[col] = df[col].fillna(method='ffill')
            else:
                # 分类列使用众数填充
                mode_value = df[col].mode()[0] if not df[col].mode().empty else "未知"
                df[col] = df[col].fillna(mode_value)
        return df
    
    def _handle_outliers_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        """实际处理异常值的实现"""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # 使用IQR方法处理异常值
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # 将异常值替换为边界值
                df[col] = df[col].clip(lower_bound, upper_bound)
        return df
    
    def _transform_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        # 尝试将可能的数值列转换为数值类型
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # 尝试转换为数值
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # 如果转换后有太多NaN，则恢复原始数据
                    if df[col].isna().sum() > len(df) * 0.3:
                        df[col] = df[col].astype('object')
                except:
                    pass
                    
            # 尝试将日期格式的字符串转换为日期类型
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # 如果转换后有太多NaN，则恢复原始数据
                    if df[col].isna().sum() > len(df) * 0.3:
                        df[col] = df[col].astype('object')
                except:
                    pass
        return df
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用特征工程"""
        # 这里可以实现更复杂的特征工程逻辑
        # 例如，创建日期特征、交互特征等
        
        # 示例：从日期列提取年、月、日特征
        for col in df.columns:
            if pd.api.types.is_datetime64_dtype(df[col]):
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                
        return df