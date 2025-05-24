# routes/api_routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
import os
import shutil
import uuid
from typing import Dict, Any
import pandas as pd
from agents.master_agent import MasterAgent
from config.config import OUTPUT_DIR

# 创建路由器
router = APIRouter()

# 创建上传目录
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 初始化主Agent
master_agent = MasterAgent()

# 存储任务状态
tasks_status = {}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, analysis_request: str = ""):
    """上传Excel文件并开始分析"""
    # 验证文件类型
    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="只支持Excel和CSV文件")
    
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    
    # 保存上传的文件
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 初始化任务状态
    tasks_status[task_id] = {
        "status": "pending",
        "file_path": file_path,
        "file_name": file.filename,
        "results": None,
        "analysis_request": analysis_request
    }
    
    # 在后台执行分析任务
    if background_tasks:
        background_tasks.add_task(process_file, task_id, file_path)
    else:
        # 同步处理（用于测试）
        process_file(task_id, file_path)
    
    return {"task_id": task_id, "status": "pending", "message": "文件上传成功，开始分析"}

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks_status[task_id]

@router.get("/results/{task_id}")
async def get_task_results(task_id: str):
    """获取任务结果"""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks_status[task_id]
    if task["status"] != "completed":
        return {"status": task["status"], "message": "任务尚未完成"}
    
    return {
        "status": "completed",
        "file_name": task["file_name"],
        "summary": task["results"].get("summary", ""),
        "report_path": task["results"].get("report_path", ""),
        "visualization_paths": task["results"].get("visualization_paths", [])
    }

def process_file(task_id: str, file_path: str):
    """处理上传的文件"""
    try:
        # 更新任务状态
        tasks_status[task_id]["status"] = "processing"
        
        # 调用主Agent处理数据
        results = master_agent.process_data_file(file_path)
        
        # 提取关键结果
        processed_results = {
            "summary": "",
            "report_path": "",
            "visualization_paths": []
        }
        
        # 提取报告路径
        if "report_results" in results and "word_report_path" in results["report_results"]:
            processed_results["report_path"] = results["report_results"]["word_report_path"]
        
        # 提取摘要
        if "report_results" in results and "summary" in results["report_results"]:
            processed_results["summary"] = results["report_results"]["summary"]
        
        # 提取可视化图表路径
        if "visualization_results" in results:
            vis_results = results["visualization_results"]
            
            # 收集所有图表路径
            for category in ["histograms", "box_plots", "category_plots", "time_series_plots"]:
                if category in vis_results:
                    for path in vis_results[category].values():
                        processed_results["visualization_paths"].append(path)
            
            # 添加单个图表
            for single_chart in ["correlation_heatmap", "scatter_matrix"]:
                if single_chart in vis_results:
                    processed_results["visualization_paths"].append(vis_results[single_chart])
        
        # 更新任务状态为完成
        tasks_status[task_id]["status"] = "completed"
        tasks_status[task_id]["results"] = processed_results
        
    except Exception as e:
        # 更新任务状态为失败
        tasks_status[task_id]["status"] = "failed"
        tasks_status[task_id]["error"] = str(e)