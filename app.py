# ====================== 1. 标准库导入（Python自带） ======================
import numpy as np
import os
import uuid
import logging
import time
from io import BytesIO
from typing import List, Dict, Any
import pathlib

# ====================== 2. 第三方库导入（pip安装） ======================
# FastAPI相关（核心Web框架）
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
# Pydantic配置（读取.env）
from pydantic_settings import BaseSettings
# PDF处理
from pypdf import PdfReader
# 向量数据库
import chromadb
# 异步HTTP请求（替代requests）
import httpx
# 加载.env文件（关键：补充你缺失的导入）
from dotenv import load_dotenv

# ====================== 3. 加载环境变量（必须在导入后、逻辑前） ======================
load_dotenv()  # 自动读取项目根目录的.env文件

# ====================== 4. 初始化核心对象（导入后立即初始化） ======================
# Configure logging - 优化日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 修复：环境变量配置（移除占位符，强制从.env读取）
class Settings(BaseSettings):
    host: str = "0.0.0.0"  # 改为0.0.0.0，支持外网访问
    port: int = 8000
    minimax_api_key: str  # 强制要求配置，无默认值
    minimax_group_id: str  # 强制要求配置，无默认值
    chunk_size: int = 500  # 文档分块大小
    chunk_overlap: int = 50  # 分块重叠长度
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    minimax_embed_model: str = "embo-01"  # 向量化模型，可通过.env覆盖

    # 缩进修正：属于Settings的内部类
    class Config:
        env_file = ".env"  # 从.env文件读取配置
        env_file_encoding = "utf-8"

# 加载配置（无配置会直接报错，提示用户创建.env）
try:
    settings = Settings()
except Exception as e:
    logger.error(f"配置加载失败：{e}")
    logger.error("请创建.env文件，包含以下内容：")
    logger.error("MINIMAX_API_KEY=你的MiniMax API密钥")
    logger.error("MINIMAX_GROUP_ID=你的MiniMax Group ID")
    raise SystemExit(1)

# 从配置类赋值（替代硬编码）
API_KEY = settings.minimax_api_key
GROUP_ID = settings.minimax_group_id

# 初始化FastAPI应用
app = FastAPI(
    title="RAG文档问答系统",
    description="基于MiniMax的RAG文档问答系统",
    version="1.0.0"
)

# 向量数据库初始化（异步友好）
DB_DIR = pathlib.Path("./chroma_db")
DB_DIR.mkdir(exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

# 获取或创建集合
try:
    collection = chroma_client.get_collection(name="documents")
    logger.info("成功连接到现有向量集合")
except Exception as e:
    logger.warning(f"创建新向量集合：{e}")
    collection = chroma_client.create_collection(
        name="documents",
        metadata={"description": "RAG文档问答系统的文档向量集合"}
    )

# 上传目录初始化
UPLOAD_DIR = pathlib.Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md'}

# 全局异步HTTP客户端（复用连接，提升性能）
async_client = httpx.AsyncClient(timeout=30)

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    将长文本拆分为重叠的小片段（RAG最佳实践）
    :param text: 原始文本
    :param chunk_size: 每个片段的长度
    :param chunk_overlap: 片段间重叠长度
    :return: 分块后的文本列表
    """
    if len(text) <= chunk_size:
        return [text.strip()]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # 跳过空片段
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    
    logger.info(f"文本分块完成：{len(chunks)}个片段，原长度{len(text)}字符")
    return chunks

def get_embedding(text: str) -> List[float]:
    """
    获取文本的向量表示
    """
    if not text.strip():
        raise ValueError("输入文本不能为空")
    
    api_key = os.getenv("MINIMAX_API_KEY")
    group_id = os.getenv("MINIMAX_GROUP_ID", "2027260839349723880")
    
    if not api_key:
        raise RuntimeError("未配置 MINIMAX_API_KEY 环境变量")
    
    url = f"https://api.minimax.chat/v1/embeddings?GroupId={group_id}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "embo-01",
        "texts": [text],
        "type": "query"
    }

    print(f"请求向量化，文本长度：{len(text)}字符")
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        resp_json = response.json()
        print(f"MiniMax返回: {resp_json}")
        
        if resp_json.get("base_resp", {}).get("status_code") != 0:
            error_msg = resp_json.get("base_resp", {}).get("status_msg", "未知错误")
            raise RuntimeError(f"MiniMax API返回错误：{error_msg}")
        
        embedding = resp_json["vectors"][0]
        
        print(f"向量维度: {len(embedding)}")
        print(f"向量前5个值: {embedding[:5]}")
        
        return embedding
    
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP请求失败：{e.response.status_code} {e.response.text}") from e
    except Exception as e:
        raise RuntimeError(f"向量化失败：{str(e)}") from e

async def chat_with_minimax(question: str, context: str) -> str:
    logger.info(f"调用 chat_with_minimax，问题：{question[:50]}，上下文长度：{len(context)}")
    """异步调用MiniMax对话接口（修复+优化）"""
    if not question or not context:
        return "错误：问题或上下文不能为空"
    
    url = f"https://api.minimax.chat/v1/text/chatcompletion?GroupId={GROUP_ID}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 上下文长度检查+日志
    context_length = len(context)
    if context_length > 5000:
        logger.warning(f"上下文过长（{context_length}字符），截断至5000字符")
        truncated_context = context[:5000]
    else:
        truncated_context = context
        logger.info(f"请求对话生成，上下文长度：{context_length}字符")
    
    messages = [
        {
            "role": "system",
            "content": """你是一个专业的文档问答助手，严格遵守以下规则：
1. 仅使用提供的上下文信息回答问题
2. 如果上下文没有相关信息，明确说明"无法从上传的文档中找到相关答案"
3. 回答要简洁、准确、易于理解
4. 保持回答的客观性，不添加额外猜测内容

上下文：{context}""".format(context=truncated_context)
        },
        {"role": "user", "content": question}
    ]
    
    payload = {
        "model": "abab6.5s-chat",
        "messages": messages,
        "tokens_to_generate": 1000,
        "temperature": 0.3,
        "top_p": 0.9
    }
    
    try:
        response = await async_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        logger.info(f"模型返回结果：{result}")
        
        # 兼容不同返回格式
        if "reply" in result:
            return result["reply"].strip()
        elif "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        elif "text" in result:
            return result["text"].strip()
        else:
            logger.error(f"未知的返回格式：{result}")
            return "无法解析模型返回结果，请检查日志"
            
    except httpx.TimeoutException:
        logger.error("对话请求超时")
        return "生成答案超时，请稍后重试"
    except httpx.HTTPStatusError as e:
        logger.error(f"对话请求失败（{e.response.status_code}）：{e.response.text}")
        return f"生成答案失败（接口错误：{e.response.status_code}）"
    except Exception as e:
        logger.error(f"对话接口异常：{str(e)}")
        return f"生成答案失败（系统错误）：{str(e)[:100]}..."

def validate_file(filename: str) -> bool:
    """验证文件类型"""
    ext = pathlib.Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

def extract_text_from_file(content: bytes, filename: str) -> str:
    """提取文件文本（支持多编码+PDF）"""
    ext = pathlib.Path(filename).suffix.lower()
    text = ""
    
    try:
        if ext == ".pdf":
            reader = PdfReader(BytesIO(content))
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    logger.info(f"PDF第{page_num}页提取文本长度：{len(page_text)}字符")
        
        elif ext in [".txt", ".md"]:
            # 优先尝试UTF-8，失败则GBK，最后容错
            encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    logger.info(f"文件解码成功（编码：{encoding}）")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = content.decode("utf-8", errors="ignore")
                logger.warning("所有编码尝试失败，使用容错解码")
        
        else:
            raise ValueError(f"不支持的文件类型：{ext}")
            
    except Exception as e:
        raise RuntimeError(f"文件解析失败：{str(e)}")
    
    return text.strip()

@app.post("/upload", summary="上传文件并向量化存储", response_description="上传结果")
async def upload_file(file: UploadFile = File(..., description="支持PDF/TXT/MD格式，最大10MB")):
    """
    上传文档并进行向量化存储：
    1. 验证文件类型和大小
    2. 提取文本内容
    3. 长文本自动分块
    4. 每个块独立向量化并存储
    5. 返回上传结果和文档ID列表
    """
    try:
        # 1. 验证文件类型
        if not validate_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型：{file.filename}，仅支持 {list(ALLOWED_EXTENSIONS)}"
            )
        
        # 2. 读取文件内容
        contents = await file.read()
        logger.info(f"开始处理文件：{file.filename}，文件大小：{len(contents)}字节")
        
        # 3. 验证文件大小
        if len(contents) > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制（最大{settings.max_file_size/1024/1024}MB）"
            )
        
        if not contents:
            raise HTTPException(status_code=400, detail="文件内容为空")
        
        # 4. 提取文本
        text = extract_text_from_file(contents, file.filename)
        if not text:
            raise HTTPException(status_code=400, detail="文件解析后内容为空")
        
        # 5. 文本分块
        chunks = split_text_into_chunks(
            text, 
            settings.chunk_size, 
            settings.chunk_overlap
        )
        
        # 6. 批量向量化+存储
        doc_ids = []
        base_doc_id = str(uuid.uuid4())
        upload_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存原始文件
        file_save_path = UPLOAD_DIR / f"{base_doc_id}_{pathlib.Path(file.filename).name}"
        with open(file_save_path, "wb") as f:
            f.write(contents)
        logger.info(f"原始文件保存至：{file_save_path}")
        
        # 逐个处理分块
        for chunk_idx, chunk in enumerate(chunks):
            # 生成向量
            emb = get_embedding(chunk)  #
            
            # 生成唯一ID（基础ID+分块序号）
            chunk_id = f"{base_doc_id}_chunk_{chunk_idx}"
            doc_ids.append(chunk_id)
            
            # 存储到向量库
            collection.add(
                documents=[chunk],
                ids=[chunk_id],
                metadatas=[{
                    "filename": file.filename,
                    "file_path": str(file_save_path),
                    "upload_time": upload_time,
                    "chunk_idx": chunk_idx,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks),
                    "base_doc_id": base_doc_id
                }],
                embeddings=[emb],
            )
        
        # 7. 返回结果
        return {
            "success": True,
            "base_doc_id": base_doc_id,
            "chunk_ids": doc_ids,
            "filename": file.filename,
            "total_chunks": len(chunks),
            "total_text_length": len(text),
            "message": f"文件处理成功！共拆分为{len(chunks)}个片段并完成向量化"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"文件上传失败：{str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"上传失败：{str(e)[:200]}..."  # 限制错误信息长度
        )

@app.get("/ask", summary="基于文档回答问题", response_description="问答结果")
async def ask_question(
    q: str = Query(..., min_length=1, max_length=500, description="你的问题"),
    top_k: int = Query(3, ge=1, le=10, description="返回的相似文档数")
):
    """
    基于已上传的文档回答问题：
    1. 问题向量化
    2. 查询相似文档片段
    3. 拼接上下文生成答案
    4. 返回答案+来源+相似度
    """
    try:
        logger.info(f"处理问答请求：{q[:50]}... (top_k={top_k})")
        
        # 1. 问题向量化
        q_emb =  get_embedding(q)
        
        # 2. 查询相似文档
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 3. 处理无结果场景
        if not results["documents"][0]:
            return {
                "success": True,
                "answer": "未找到相关文档，请先上传包含相关信息的文件",
                "question": q,
                "sources": [],
                "similarity_scores": []
            }
        
        # 4. 整理结果（去重+计算相似度）
        contexts = []
        sources = []
        similarity_scores = []
        
        # Chroma的distance越小越相似，转换为0-1的相似度得分
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            contexts.append(doc)
            sources.append(meta.get("filename", "未知文件"))
            # 距离通常在0-2之间，转换为相似度（1 - 距离/2）
            similarity = max(0.0, min(1.0, 1 - dist/2))
            similarity_scores.append(round(similarity, 4))
        
        # 5. 去重来源（保留顺序）
        unique_sources = []
        seen = set()
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        
        # 6. 拼接上下文
        context = "\n\n--- 分割线 ---\n\n".join(contexts)
        
        # 7. 生成答案
        answer = await chat_with_minimax(q, context)
        
        # 8. 返回结构化结果
        return {
            "success": True,
            "answer": answer,
            "question": q,
            "sources": unique_sources,
            "similarity_scores": similarity_scores,
            "matched_chunks": len(contexts),
            "hint": "相似度得分越接近1，文档相关性越高"
        }
        
    except Exception as e:
        logger.error(f"问答请求失败：{str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"提问失败：{str(e)[:200]}..."
        )

@app.get("/documents", summary="列出所有已上传文档", response_description="文档列表")
async def list_documents():
    """获取所有已上传文档的元数据（去重）"""
    try:
        results = collection.get(include=["metadatas"])
        
        # 按base_doc_id分组，去重
        doc_map: Dict[str, Dict[str, Any]] = {}
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
            base_id = meta.get("base_doc_id", doc_id)
            if base_id not in doc_map:
                doc_map[base_id] = {
                    "base_doc_id": base_id,
                    "filename": meta.get("filename", "未知文件"),
                    "upload_time": meta.get("upload_time", "未知时间"),
                    "total_chunks": meta.get("total_chunks", 1),
                    "file_path": meta.get("file_path", ""),
                    "chunk_ids": []
                }
            doc_map[base_id]["chunk_ids"].append(doc_id)
        
        # 转换为列表
        documents = list(doc_map.values())
        
        return {
            "success": True,
            "total_documents": len(documents),
            "total_chunks": len(results["ids"]),
            "documents": documents
        }
        
    except Exception as e:
        logger.error(f"获取文档列表失败：{str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"获取文档列表失败：{str(e)[:200]}..."
        )

@app.delete("/documents/{doc_id}", summary="删除文档", response_description="删除结果")
async def delete_document(doc_id: str, delete_all_chunks: bool = True):
    """
    删除文档：
    - delete_all_chunks=True：删除该base_doc_id下的所有分块
    - delete_all_chunks=False：仅删除指定的chunk_id
    """
    try:
        if delete_all_chunks:
            # 获取该base_doc_id下的所有分块
            results = collection.get(include=["metadatas"])
            chunk_ids_to_delete = []
            for cid, meta in zip(results["ids"], results["metadatas"]):
                if meta.get("base_doc_id") == doc_id or cid == doc_id:
                    chunk_ids_to_delete.append(cid)
            
            if not chunk_ids_to_delete:
                raise HTTPException(status_code=404, detail="未找到该文档的任何分块")
            
            # 删除分块
            collection.delete(ids=chunk_ids_to_delete)
            
            # 删除原始文件（如果存在）
            for meta in results["metadatas"]:
                if meta.get("base_doc_id") == doc_id and "file_path" in meta:
                    file_path = pathlib.Path(meta["file_path"])
                    if file_path.exists():
                        file_path.unlink()
                        logger.info(f"删除原始文件：{file_path}")
                    break
            
            logger.info(f"删除文档{doc_id}的{len(chunk_ids_to_delete)}个分块")
            return {
                "success": True,
                "message": f"成功删除文档{doc_id}的{len(chunk_ids_to_delete)}个分块",
                "deleted_chunk_ids": chunk_ids_to_delete
            }
        
        else:
            # 仅删除指定chunk_id
            if doc_id not in collection.get(ids=[doc_id])["ids"]:
                raise HTTPException(status_code=404, detail="未找到该分块ID")
            
            collection.delete(ids=[doc_id])
            logger.info(f"删除单个分块：{doc_id}")
            return {
                "success": True,
                "message": f"成功删除分块{doc_id}",
                "deleted_chunk_ids": [doc_id]
            }
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"删除文档失败：{str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"删除文档失败：{str(e)[:200]}..."
        )

@app.get("/health", summary="健康检查", response_description="服务状态")
async def health_check():
    """服务健康检查接口"""
    return {
        "status": "healthy",
        "service": "RAG文档问答系统",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "minimax_group_id": GROUP_ID[:10] + "..." if GROUP_ID else "未配置",
            "minimax_embed_model": settings.minimax_embed_model,
            "chunk_size": settings.chunk_size,
            "max_file_size_mb": settings.max_file_size / 1024 / 1024
        },
        "collections": {
            "name": "documents",
            "count": collection.count()
        }
    }

@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    logger.info("="*60)
    logger.info("RAG文档问答系统启动成功！")
    logger.info(f"API文档地址：http://{settings.host}:{settings.port}/docs")
    logger.info(f"健康检查地址：http://{settings.host}:{settings.port}/health")
    logger.info(f"数据存储目录：{DB_DIR.absolute()}")
    logger.info(f"文件上传目录：{UPLOAD_DIR.absolute()}")
    logger.info(f"当前向量数量：{collection.count()}")
    logger.info("="*60)

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭事件"""
    await async_client.aclose()
    logger.info("异步HTTP客户端已关闭")
    logger.info("RAG文档问答系统已停止")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
        access_log=True
    )