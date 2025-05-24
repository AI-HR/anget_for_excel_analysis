# agents/visualization_agent.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os
from typing import Dict, Any, List, Optional, Tuple
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config.config import OPENAI_API_KEY, MODEL_NAME, MODEL_BASE_URL, OUTPUT_DIR

# 设置中文字体支持
# 在Mac系统上，常用的中文字体包括'Arial Unicode MS', 'Heiti SC', 'PingFang SC'
matplotlib.rcParams['font.family'] = ['Arial Unicode MS',  'sans-serif']
# 解决负号显示问题
matplotlib.rcParams['axes.unicode_minus'] = False

class VisualizationAgent:
    """基于LLM的智能数据可视化Agent"""
    
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
        
        # 可视化输出目录
        self.output_dir = os.path.join(OUTPUT_DIR, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置图表保存参数，确保中文正常显示
        self.save_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight'
        }
        
        # 定义可视化工具
        self.tools = [
            Tool(
                name="柱状图",
                func=self._bar_chart,
                description="创建柱状图，适合比较不同类别的数值"
            ),
            Tool(
                name="折线图",
                func=self._line_chart,
                description="创建折线图，适合展示时间序列或趋势"
            ),
            Tool(
                name="散点图",
                func=self._scatter_plot,
                description="创建散点图，适合展示两个变量之间的关系"
            ),
            Tool(
                name="箱线图",
                func=self._box_plot,
                description="创建箱线图，适合展示数据分布和异常值"
            ),
            Tool(
                name="热力图",
                func=self._heatmap,
                description="创建热力图，适合展示变量之间的相关性"
            ),
            Tool(
                name="饼图",
                func=self._pie_chart,
                description="创建饼图，适合展示部分与整体的关系"
            ),
            Tool(
                name="直方图",
                func=self._histogram,
                description="创建直方图，适合展示数值分布"
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
    
    def visualize_data(self, df: pd.DataFrame, analysis_results: Dict[str, Any],
                       business_context: Optional[Dict[str, Any]] = None,
                       analysis_themes: Optional[List[Dict[str, Any]]] = None,
                       analysis_outline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """根据分析结果创建可视化图表"""
        # 设置当前数据框
        self.current_df = df

        # 数据概览
        data_info = self._get_data_overview(df)

        # 构建Agent提示
        prompt = f"""
        我需要为一个数据集创建可视化图表。数据集信息如下：
        
        {data_info}
        
        分析结果概要：
        {self._format_analysis_results(analysis_results)}
        
        业务场景信息：
        {business_context.get('context', '无业务场景信息') if business_context else '无业务场景信息'}
        
        分析主题：
        {', '.join([theme['name'] for theme in analysis_themes]) if analysis_themes else '无分析主题'}
        
        请根据以下要求执行任务：
        1. 确保所有可视化图表严格遵循报告的主题和业务场景。
        2. 选择最适合展示数据特征和分析结果的可视化方法。
        3. 创建相应的可视化图表，并结合业务场景和分析主题提供解读。
        4. 确保图表清晰、信息丰富且易于理解。
        
        请给出详细的可视化计划和每一步的理由。
        """

        # 运行Agent获取可视化计划
        try:
            visualization_plan = self.agent.invoke(prompt)
            # 执行可视化计划
            visualization_results = self._execute_visualization_plan(df, analysis_results, visualization_plan)
            return visualization_results
        except Exception as e:
            print(f"创建可视化图表时出错: {str(e)}")
            # 如果Agent调用失败，执行基本的可视化
            return self._basic_visualization(df, analysis_results)
    
    def _get_data_overview(self, df: pd.DataFrame) -> str:
        """获取数据概览信息"""
        overview = f"行数: {df.shape[0]}, 列数: {df.shape[1]}\n"
        overview += f"列名: {', '.join(df.columns.tolist())}\n"
        overview += "数据类型:\n"
        
        for col in df.columns:
            dtype = df[col].dtype
            overview += f"- {col}: {dtype}\n"
                
        return overview
    
    def _format_analysis_results(self, analysis_results: Dict[str, Any]) -> str:
        """格式化分析结果，用于提示"""
        formatted = ""
        
        if "descriptive" in analysis_results:
            formatted += "描述性统计分析结果可用\n"
            
        if "correlation" in analysis_results:
            formatted += "相关性分析结果可用\n"
            if "high_correlation_pairs" in analysis_results["correlation"]:
                pairs = analysis_results["correlation"]["high_correlation_pairs"]
                if pairs:
                    formatted += f"发现{len(pairs)}对高相关性特征\n"
        
        if "distribution" in analysis_results:
            formatted += "分布分析结果可用\n"
        
        if "anova" in analysis_results:
            formatted += "方差分析结果可用\n"
        
        if "time_series" in analysis_results:
            formatted += "时间序列分析结果可用\n"
        
        if "clustering" in analysis_results:
            formatted += "聚类分析结果可用\n"
        
        return formatted
    
    def _execute_visualization_plan(self, df: pd.DataFrame, analysis_results: Dict[str, Any], visualization_plan: Any) -> Dict[str, Any]:
        """执行可视化计划"""
        results = {}
        
        # 根据数据类型和分析结果创建基本可视化
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # 创建描述性统计可视化
        if numeric_cols:
            # 创建直方图和箱线图
            for col in numeric_cols[:min(5, len(numeric_cols))]:
                hist_path = self._create_histogram(df, col)
                if hist_path:
                    results.setdefault("histograms", {})[col] = hist_path
                
                box_path = self._create_box_plot(df, col)
                if box_path:
                    results.setdefault("box_plots", {})[col] = box_path
        
        # 创建相关性热力图
        if len(numeric_cols) >= 2 and "correlation" in analysis_results:
            heatmap_path = self._create_correlation_heatmap(df, numeric_cols)
            if heatmap_path:
                results["correlation_heatmap"] = heatmap_path
        
        # 创建类别计数图
        if categorical_cols:
            for col in categorical_cols[:min(3, len(categorical_cols))]:
                bar_path = self._create_category_count_plot(df, col)
                if bar_path:
                    results.setdefault("category_plots", {})[col] = bar_path
        
        # 创建散点图矩阵
        if len(numeric_cols) >= 2:
            scatter_matrix_path = self._create_scatter_matrix(df, numeric_cols[:min(4, len(numeric_cols))])
            if scatter_matrix_path:
                results["scatter_matrix"] = scatter_matrix_path
        
        # 创建时间序列图
        if date_cols and numeric_cols:
            for date_col in date_cols[:1]:
                for numeric_col in numeric_cols[:min(3, len(numeric_cols))]:
                    line_path = self._create_time_series_plot(df, date_col, numeric_col)
                    if line_path:
                        results.setdefault("time_series_plots", {})[
                            f"{date_col}_{numeric_col}"] = line_path
        
        return results
    
    def _basic_visualization(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建基本的可视化图表"""
        # 与_execute_visualization_plan相同，但不依赖于Agent的计划
        return self._execute_visualization_plan(df, analysis_results, None)
    
    # 工具函数实现
    def _bar_chart(self, query: str) -> str:
        """创建柱状图"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        try:
            # 解析查询，获取要绘制的列
            import re
            cols = re.findall(r'列[：:](\w+)', query)
            if not cols:
                categorical_cols = self.current_df.select_dtypes(include=['object', 'category']).columns.tolist()
                if not categorical_cols:
                    return "错误：未找到合适的分类列"
                col = categorical_cols[0]
            else:
                col = cols[0]
                if col not in self.current_df.columns:
                    return f"错误：列 '{col}' 不存在"
            
            # 创建柱状图
            chart_path = self._create_category_count_plot(self.current_df, col)
            if chart_path:
                return f"已创建柱状图，展示了 '{col}' 列的类别分布。图表保存在: {chart_path}"
            else:
                return "创建柱状图失败"
        except Exception as e:
            return f"创建柱状图时出错: {str(e)}"
    
    def _line_chart(self, query: str) -> str:
        """创建折线图"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        try:
            # 解析查询，获取要绘制的列
            import re
            date_cols = re.findall(r'日期列[：:](\w+)', query)
            value_cols = re.findall(r'值列[：:](\w+)', query)
            
            # 查找日期列
            if not date_cols:
                date_columns = self.current_df.select_dtypes(include=['datetime']).columns.tolist()
                if not date_columns:
                    # 尝试将字符串列转换为日期
                    for col in self.current_df.select_dtypes(include=['object']).columns:
                        try:
                            pd.to_datetime(self.current_df[col])
                            date_columns.append(col)
                        except:
                            pass
                if not date_columns:
                    return "错误：未找到合适的日期列"
                date_col = date_columns[0]
            else:
                date_col = date_cols[0]
                if date_col not in self.current_df.columns:
                    return f"错误：列 '{date_col}' 不存在"
            
            # 查找数值列
            if not value_cols:
                numeric_cols = self.current_df.select_dtypes(include=['number']).columns.tolist()
                if not numeric_cols:
                    return "错误：未找到合适的数值列"
                value_col = numeric_cols[0]
            else:
                value_col = value_cols[0]
                if value_col not in self.current_df.columns:
                    return f"错误：列 '{value_col}' 不存在"
            
            # 创建折线图
            chart_path = self._create_time_series_plot(self.current_df, date_col, value_col)
            if chart_path:
                return f"已创建折线图，展示了 '{date_col}' 和 '{value_col}' 的时间序列关系。图表保存在: {chart_path}"
            else:
                return "创建折线图失败"
        except Exception as e:
            return f"创建折线图时出错: {str(e)}"
    
    def _scatter_plot(self, query: str) -> str:
        """创建散点图"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        try:
            # 解析查询，获取要绘制的列
            import re
            x_cols = re.findall(r'X列[：:](\w+)', query)
            y_cols = re.findall(r'Y列[：:](\w+)', query)
            
            # 查找数值列
            numeric_cols = self.current_df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) < 2:
                return "错误：需要至少两个数值列才能创建散点图"
            
            if not x_cols:
                x_col = numeric_cols[0]
            else:
                x_col = x_cols[0]
                if x_col not in self.current_df.columns:
                    return f"错误：列 '{x_col}' 不存在"
            
            if not y_cols:
                y_col = numeric_cols[1] if numeric_cols[0] == x_col and len(numeric_cols) > 1 else numeric_cols[0]
            else:
                y_col = y_cols[0]
                if y_col not in self.current_df.columns:
                    return f"错误：列 '{y_col}' 不存在"
            
            # 创建散点图
            chart_path = self._create_scatter_plot(self.current_df, x_col, y_col)
            if chart_path:
                return f"已创建散点图，展示了 '{x_col}' 和 '{y_col}' 的关系。图表保存在: {chart_path}"
            else:
                return "创建散点图失败"
        except Exception as e:
            return f"创建散点图时出错: {str(e)}"
    
    def _box_plot(self, query: str) -> str:
        """创建箱线图"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        try:
            # 解析查询，获取要绘制的列
            import re
            cols = re.findall(r'列[：:](\w+)', query)
            
            if not cols:
                numeric_cols = self.current_df.select_dtypes(include=['number']).columns.tolist()
                if not numeric_cols:
                    return "错误：未找到合适的数值列"
                col = numeric_cols[0]
            else:
                col = cols[0]
                if col not in self.current_df.columns:
                    return f"错误：列 '{col}' 不存在"
            
            # 创建箱线图
            chart_path = self._create_box_plot(self.current_df, col)
            if chart_path:
                return f"已创建箱线图，展示了 '{col}' 的分布。图表保存在: {chart_path}"
            else:
                return "创建箱线图失败"
        except Exception as e:
            return f"创建箱线图时出错: {str(e)}"
    
    def _heatmap(self, query: str) -> str:
        """创建热力图"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        try:
            # 获取数值列
            numeric_cols = self.current_df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) < 2:
                return "错误：需要至少两个数值列才能创建热力图"
            
            # 创建热力图
            chart_path = self._create_correlation_heatmap(self.current_df, numeric_cols)
            if chart_path:
                return f"已创建热力图，展示了数值列之间的相关性。图表保存在: {chart_path}"
            else:
                return "创建热力图失败"
        except Exception as e:
            return f"创建热力图时出错: {str(e)}"
    
    def _pie_chart(self, query: str) -> str:
        """创建饼图"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        try:
            # 解析查询，获取要绘制的列
            import re
            cols = re.findall(r'列[：:](\w+)', query)
            
            if not cols:
                categorical_cols = self.current_df.select_dtypes(include=['object', 'category']).columns.tolist()
                if not categorical_cols:
                    return "错误：未找到合适的分类列"
                col = categorical_cols[0]
            else:
                col = cols[0]
                if col not in self.current_df.columns:
                    return f"错误：列 '{col}' 不存在"
            
            # 创建饼图
            chart_path = self._create_pie_chart(self.current_df, col)
            if chart_path:
                return f"已创建饼图，展示了 '{col}' 的类别分布。图表保存在: {chart_path}"
            else:
                return "创建饼图失败"
        except Exception as e:
            return f"创建饼图时出错: {str(e)}"
    
    def _histogram(self, query: str) -> str:
        """创建直方图"""
        if self.current_df is None:
            return "错误：没有可用的数据框"
        
        try:
            # 解析查询，获取要绘制的列
            import re
            cols = re.findall(r'列[：:](\w+)', query)
            
            if not cols:
                numeric_cols = self.current_df.select_dtypes(include=['number']).columns.tolist()
                if not numeric_cols:
                    return "错误：未找到合适的数值列"
                col = numeric_cols[0]
            else:
                col = cols[0]
                if col not in self.current_df.columns:
                    return f"错误：列 '{col}' 不存在"
            
            # 创建直方图
            chart_path = self._create_histogram(self.current_df, col)
            if chart_path:
                return f"已创建直方图，展示了 '{col}' 的分布。图表保存在: {chart_path}"
            else:
                return "创建直方图失败"
        except Exception as e:
            return f"创建直方图时出错: {str(e)}"
    
    # 实际实现函数
    def _create_category_count_plot(self, df: pd.DataFrame, column: str) -> Optional[str]:
        """创建类别计数图"""
        try:
            plt.figure(figsize=(10, 6))
            # 确保使用中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS',  'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            value_counts = df[column].value_counts()
            # 如果类别太多，只显示前10个
            if len(value_counts) > 10:
                value_counts = value_counts.iloc[:10]
            
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f"{column} 类别分布", fontproperties='Arial Unicode MS')
            plt.xlabel(column, fontproperties='Arial Unicode MS')
            plt.ylabel("计数", fontproperties='Arial Unicode MS')
            plt.xticks(rotation=45, fontproperties='Arial Unicode MS')
            plt.tight_layout()
            
            # 保存图表
            file_path = os.path.join(self.output_dir, f"bar_chart_{column}.png")
            plt.savefig(file_path, **self.save_kwargs)
            plt.close()
            return file_path
        except Exception as e:
            print(f"创建类别计数图时出错: {str(e)}")
            return None
    
    def _create_time_series_plot(self, df: pd.DataFrame, date_column: str, value_column: str) -> Optional[str]:
        """创建时间序列图"""
        try:
            # 确保日期列是日期类型
            if not pd.api.types.is_datetime64_dtype(df[date_column]):
                df = df.copy()
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            
            # 按日期排序
            df_sorted = df.sort_values(by=date_column)
            
            plt.figure(figsize=(12, 6))
            # 确保使用中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS',  'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.plot(df_sorted[date_column], df_sorted[value_column])
            plt.title(f"{value_column} 随时间变化趋势", fontproperties='Arial Unicode MS')
            plt.xlabel(date_column, fontproperties='Arial Unicode MS')
            plt.ylabel(value_column, fontproperties='Arial Unicode MS')
            plt.grid(True)
            plt.tight_layout()
            
            # 保存图表
            file_path = os.path.join(self.output_dir, f"line_chart_{date_column}_{value_column}.png")
            plt.savefig(file_path, **self.save_kwargs)
            plt.close()
            return file_path
        except Exception as e:
            print(f"创建时间序列图时出错: {str(e)}")
            return None
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_column: str, y_column: str) -> Optional[str]:
        """创建散点图"""
        try:
            plt.figure(figsize=(10, 6))
            # 确保使用中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            sns.scatterplot(data=df, x=x_column, y=y_column)
            plt.title(f"{x_column} vs {y_column}", fontproperties='Arial Unicode MS')
            plt.xlabel(x_column, fontproperties='Arial Unicode MS')
            plt.ylabel(y_column, fontproperties='Arial Unicode MS')
            plt.grid(True)
            plt.tight_layout()
            
            # 保存图表
            file_path = os.path.join(self.output_dir, f"scatter_plot_{x_column}_{y_column}.png")
            plt.savefig(file_path, **self.save_kwargs)
            plt.close()
            return file_path
        except Exception as e:
            print(f"创建散点图时出错: {str(e)}")
            return None
    
    def _create_box_plot(self, df: pd.DataFrame, column: str) -> Optional[str]:
        """创建箱线图"""
        try:
            plt.figure(figsize=(8, 6))
            # 确保使用中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS',  'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            sns.boxplot(y=df[column])
            plt.title(f"{column} 分布", fontproperties='Arial Unicode MS')
            plt.ylabel(column, fontproperties='Arial Unicode MS')
            plt.grid(True)
            plt.tight_layout()
            
            # 保存图表
            file_path = os.path.join(self.output_dir, f"box_plot_{column}.png")
            plt.savefig(file_path, **self.save_kwargs)
            plt.close()
            return file_path
        except Exception as e:
            print(f"创建箱线图时出错: {str(e)}")
            return None
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, columns: List[str]) -> Optional[str]:
        """创建相关性热力图"""
        try:
            # 计算相关系数矩阵
            corr_matrix = df[columns].corr()
            
            plt.figure(figsize=(10, 8))
            # 确保使用中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS',  'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title("相关性热力图", fontproperties='Arial Unicode MS')
            plt.tight_layout()
            
            # 保存图表
            file_path = os.path.join(self.output_dir, "correlation_heatmap.png")
            plt.savefig(file_path, **self.save_kwargs)
            plt.close()
            return file_path
        except Exception as e:
            print(f"创建相关性热力图时出错: {str(e)}")
            return None
    
    def _create_pie_chart(self, df: pd.DataFrame, column: str) -> Optional[str]:
        """创建饼图"""
        try:
            value_counts = df[column].value_counts()
            # 如果类别太多，只显示前5个，其余归为"其他"
            if len(value_counts) > 5:
                top_5 = value_counts.iloc[:5]
                others = pd.Series([value_counts.iloc[5:].sum()], index=['其他'])
                value_counts = pd.concat([top_5, others])
            
            plt.figure(figsize=(10, 8))
            # 确保使用中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 为标签设置字体属性
            font_props = {'fontproperties': 'Arial Unicode MS'}
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, textprops=font_props)
            plt.title(f"{column} 类别分布", fontproperties='Arial Unicode MS')
            plt.axis('equal')  # 确保饼图是圆形的
            
            # 保存图表
            file_path = os.path.join(self.output_dir, f"pie_chart_{column}.png")
            plt.savefig(file_path, **self.save_kwargs)
            plt.close()
            return file_path
        except Exception as e:
            print(f"创建饼图时出错: {str(e)}")
            return None
    
    def _create_histogram(self, df: pd.DataFrame, column: str) -> Optional[str]:
        """创建直方图"""
        try:
            plt.figure(figsize=(10, 6))
            # 确保使用中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS',  'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            sns.histplot(df[column], kde=True)
            plt.title(f"{column} 分布", fontproperties='Arial Unicode MS')
            plt.xlabel(column, fontproperties='Arial Unicode MS')
            plt.ylabel("频率", fontproperties='Arial Unicode MS')
            plt.grid(True)
            plt.tight_layout()
            
            # 保存图表
            file_path = os.path.join(self.output_dir, f"histogram_{column}.png")
            plt.savefig(file_path, **self.save_kwargs)
            plt.close()
            return file_path
        except Exception as e:
            print(f"创建直方图时出错: {str(e)}")
            return None
    
    def _create_scatter_matrix(self, df: pd.DataFrame, columns: List[str]) -> Optional[str]:
        """创建散点图矩阵"""
        try:
            # 限制列数，避免图表过于复杂
            if len(columns) > 4:
                columns = columns[:4]
            
            # 确保使用中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
                
            # 使用seaborn的pairplot创建散点图矩阵
            g = sns.pairplot(df[columns], diag_kind='kde', height=2.5)
            g.fig.suptitle("散点图矩阵", y=1.02, fontproperties='Arial Unicode MS')
            
            # 保存图表
            file_path = os.path.join(self.output_dir, "scatter_matrix.png")
            plt.savefig(file_path, **self.save_kwargs)
            plt.close()
            return file_path
        except Exception as e:
            print(f"创建散点图矩阵时出错: {str(e)}")
            return None