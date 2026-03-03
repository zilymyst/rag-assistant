# rag-assistant
基于MiniMax的RAG文档问答系统
基于 MiniMax的文档问答系统，支持上传 PDF/TXT/MD，通过自然语言提问。
 快速开始
1. 克隆并安装
bash
git clone <repo>
cd rag-doc-qa
pip install fastapi pypdf chromadb httpx uvicorn python-multipart pydantic
2. 配置环境变量
创建 .env 文件：
ini
MINIMAX_API_KEY=你的API密钥
MINIMAX_GROUP_ID=你的Group ID
3. 启动服务
bash
uvicorn main:app --reload
访问 http://localhost:8000/docs 查看 API 文档

📡 API 接口
POST /upload
上传文件（PDF/TXT/MD），自动分块并向量化存储

参数：file (form-data)
返回：{"message": "...", "doc_ids": [...]}

POST /ask
基于已上传内容回答问题

请求体：{"question": "你的问题"}
返回：{"answer": "...", "sources": [...]}

⚙️ 配置说明
变量	默认值	说明
MINIMAX_API_KEY	必填	MiniMax API密钥
MINIMAX_GROUP_ID	必填	MiniMax Group ID
CHUNK_SIZE	500	分块字符数
MAX_FILE_SIZE	10MB	文件大小限制
📦 依赖
fastapi, pypdf, chromadb, httpx, uvicorn

📄 许可证
MIT