# app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uvicorn
from routes.api_routes import router as api_router
from config.config import OUTPUT_DIR

# 创建FastAPI应用
app = FastAPI(title="Excel智能分析报告生成系统", description="基于LLM的Excel数据智能分析系统")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# 设置模板目录
templates = Jinja2Templates(directory="templates")

# 注册API路由
app.include_router(api_router, prefix="/api", tags=["API"])

# 首页路由
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 下载报告路由
@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    full_path = os.path.join(OUTPUT_DIR, file_path)
    if not os.path.exists(full_path):
        return {"error": "文件不存在"}
    return FileResponse(full_path, filename=os.path.basename(full_path))

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # 如果templates目录为空，创建一个简单的index.html
    index_html_path = os.path.join("templates", "index.html")
    if not os.path.exists(index_html_path):
        with open(index_html_path, "w", encoding="utf-8") as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Excel智能分析报告生成系统</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    .container { max-width: 800px; margin-top: 50px; }
                    .result-area { margin-top: 30px; display: none; }
                    .loading { display: none; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="text-center mb-4">Excel智能分析报告生成系统</h1>
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">上传Excel文件</h5>
                            <form id="upload-form" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <input class="form-control" type="file" id="file-input" accept=".xlsx,.xls,.csv">
                                </div>
                                <button type="submit" class="btn btn-primary">开始分析</button>
                                <div class="spinner-border text-primary loading mt-3" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            </form>
                        </div>
                    </div>
                    
                    <div class="result-area" id="result-area">
                        <div class="card mt-4">
                            <div class="card-header">分析结果</div>
                            <div class="card-body">
                                <h5 class="card-title" id="result-title">分析完成</h5>
                                <p class="card-text" id="result-summary"></p>
                                <div id="result-links"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <script>
                    document.getElementById('upload-form').addEventListener('submit', async function(e) {
                        e.preventDefault();
                        
                        const fileInput = document.getElementById('file-input');
                        if (!fileInput.files[0]) {
                            alert('请选择文件');
                            return;
                        }
                        
                        const formData = new FormData();
                        formData.append('file', fileInput.files[0]);
                        
                        // 显示加载中
                        document.querySelector('.loading').style.display = 'inline-block';
                        
                        try {
                            // 上传文件
                            const response = await fetch('/api/upload', {
                                method: 'POST',
                                body: formData
                            });
                            
                            const data = await response.json();
                            
                            if (data.task_id) {
                                // 轮询任务状态
                                checkTaskStatus(data.task_id);
                            } else {
                                alert('上传失败');
                                document.querySelector('.loading').style.display = 'none';
                            }
                        } catch (error) {
                            console.error('Error:', error);
                            alert('上传失败');
                            document.querySelector('.loading').style.display = 'none';
                        }
                    });
                    
                    async function checkTaskStatus(taskId) {
                        try {
                            const response = await fetch(`/api/status/${taskId}`);
                            const data = await response.json();
                            
                            if (data.status === 'completed') {
                                // 获取结果
                                getTaskResults(taskId);
                            } else if (data.status === 'failed') {
                                alert('分析失败: ' + (data.error || '未知错误'));
                                document.querySelector('.loading').style.display = 'none';
                            } else {
                                // 继续轮询
                                setTimeout(() => checkTaskStatus(taskId), 2000);
                            }
                        } catch (error) {
                            console.error('Error:', error);
                            alert('获取任务状态失败');
                            document.querySelector('.loading').style.display = 'none';
                        }
                    }
                    
                    async function getTaskResults(taskId) {
                        try {
                            const response = await fetch(`/api/results/${taskId}`);
                            const data = await response.json();
                            
                            // 隐藏加载中
                            document.querySelector('.loading').style.display = 'none';
                            
                            // 显示结果区域
                            document.getElementById('result-area').style.display = 'block';
                            
                            // 设置结果标题
                            document.getElementById('result-title').textContent = `分析完成: ${data.file_name}`;
                            
                            // 设置摘要
                            document.getElementById('result-summary').textContent = data.summary || '无摘要';
                            
                            // 设置下载链接
                            const linksDiv = document.getElementById('result-links');
                            linksDiv.innerHTML = '';
                            
                            if (data.report_path) {
                                const reportLink = document.createElement('a');
                                reportLink.href = `/download/${data.report_path.split('/').pop()}`;
                                reportLink.className = 'btn btn-success mt-2';
                                reportLink.textContent = '下载分析报告';
                                reportLink.target = '_blank';
                                linksDiv.appendChild(reportLink);
                            }
                            
                            // 添加可视化图表链接
                            if (data.visualization_paths && data.visualization_paths.length > 0) {
                                const visualizationTitle = document.createElement('h6');
                                visualizationTitle.className = 'mt-3';
                                visualizationTitle.textContent = '可视化图表:';
                                linksDiv.appendChild(visualizationTitle);
                                
                                const row = document.createElement('div');
                                row.className = 'row';
                                linksDiv.appendChild(row);
                                
                                data.visualization_paths.forEach(path => {
                                    const col = document.createElement('div');
                                    col.className = 'col-md-4 mb-3';
                                    
                                    const img = document.createElement('img');
                                    img.src = `/output/${path.split('/').pop()}`;
                                    img.className = 'img-fluid';
                                    img.alt = '可视化图表';
                                    
                                    col.appendChild(img);
                                    row.appendChild(col);
                                });
                            }
                        } catch (error) {
                            console.error('Error:', error);
                            alert('获取任务结果失败');
                        }
                    }
                </script>
                
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            </body>
            </html>
            """)
    
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)