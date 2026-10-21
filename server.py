#作者：Narei~
#版本：1.0-20261021
#可修改config区

import asyncio
import time
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from llama_cpp import Llama
from typing import List, Optional, Union


# ---------------- config----------------
MODEL_PATH = "qwen3.5-9b.gguf"   # 换成你的实际 GGUF 路径
MODEL_NAME = "qwen3.5-9b"
N_CTX = 8192
N_GPU_LAYERS = 0
REQUEST_TIMEOUT = 180.0
PROMPT_FILE = "prom.txt"                         # 可选的外部提示词文件

# ---------------- 2. 加载外部提示词（可选） ----------------
try:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
    print(f"[ok] 已加载外部提示词 ({len(SYSTEM_PROMPT)} 字符)")
except FileNotFoundError:
    SYSTEM_PROMPT = "你是一个有帮助的AI助手。"
    print("[warn] 未找到提示词文件，使用默认提示词")

# ---------------- 3. 初始化模型 ----------------
print("[wait] 正在加载模型到内存/显存...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=True,
)
print("[ok] 模型加载完成！")

app = FastAPI(title="Nanobot-compatible Local LLM Server")

# ---------------- 4. 请求数据结构 ----------------
class Message(BaseModel):
    role: str
    content: Union[str, list, None] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    stream: Optional[bool] = False

# ---------------- 5. 内容标准化（兼容数组型 content） ----------------
def normalize_content(content: Union[str, list, None]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts).strip()

# ---------------- 6. token 计数 + 历史裁剪 ----------------
def count_tokens(text: str) -> int:
    try:
        return len(llm.tokenize(text.encode("utf-8", errors="ignore"), special=False))
    except Exception:
        return int(len(text) * 0.7)  # tokenizer 异常时按 1 token ~= 1.5 字符粗估

def total_tokens(messages: List[dict]) -> int:
    return sum(count_tokens(m.get("content", "")) + 8 for m in messages)

def trim_messages(messages: List[dict], reserve_output: int) -> List[dict]:
    """
    保留首条 system + 最近的对话轮，确保 prompt token <= N_CTX - reserve - 模板开销。
    从最旧的非 system 轮次开始丢弃；极端超长时对最后一条 content 做字符截断兜底。
    """
    budget = N_CTX - max(reserve_output, 64) - 128

    # 优先丢弃最旧的非 system 轮次
    while len(messages) > 2 and total_tokens(messages) > budget:
        dropped = False
        for i in range(1, len(messages)):
            if messages[i].get("role") != "system":
                messages.pop(i)
                dropped = True
                break
        if not dropped:
            break

    # 兜底：单条仍超长时截断最后一条内容
    if total_tokens(messages) > budget and messages:
        last = messages[-1]
        max_chars = max(int(budget * 1.5), 200)
        if len(last.get("content", "")) > max_chars:
            last["content"] = last["content"][:max_chars] + "\n...[truncated]"
    return messages


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # 仅记录 chat/completions 接口
    if "/v1/chat/completions" in request.url.path:
        body = await request.body()
        try:
            data = json.loads(body)
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "messages_count": len(data.get("messages", [])),
                "has_tools": "tools" in data,
                "stream": data.get("stream", False),
                "full_request": data
            }
            # 追加写入文件，避免覆盖
            with open("/tmp/nanobot_requests.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False, indent=2) + "\n---\n")
        except Exception as e:
            print(f"[LOG ERROR] Failed to parse request body: {e}")
    
    response = await call_next(request)
    return response
# ---------------- 7. 422 校验错误处理 ----------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("[422] 校验失败详情：")
    print(exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": now,
                "owned_by": "local-llama-cpp",
            }
        ],
    }

# ---------------- 9. 聊天补全主路由 ----------------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # 标准化消息
    messages = [
        {"role": m.role, "content": normalize_content(m.content)}
        for m in request.messages
    ]

    # 注入外部提示词
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = f"{SYSTEM_PROMPT}\n\n---\n补充要求：{messages[0]['content']}"
    else:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # 推理前裁剪，防止 exceed context window
    reserve = request.max_tokens or 512
    messages = trim_messages(messages, reserve)

    kwargs = {
        "messages": messages,
        "max_tokens": min(reserve, N_CTX - 128),
        "temperature": request.temperature,
    }
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p

    def run_inference(call_kwargs):
        return llm.create_chat_completion(**call_kwargs)

    try:

        if request.tools:
            call_kwargs = dict(kwargs)
            call_kwargs["tools"] = request.tools
            call_kwargs["tool_choice"] = request.tool_choice
        else:
            call_kwargs = kwargs


        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(run_inference, call_kwargs),
                timeout=REQUEST_TIMEOUT,
            )
        except TypeError as e:
            print("[warn] tools 参数不受支持，退回纯聊天：", e)
            response = await asyncio.wait_for(
                asyncio.to_thread(run_inference, kwargs),
                timeout=REQUEST_TIMEOUT,
            )

    except asyncio.TimeoutError:
        print(f"[error] 推理超时（{REQUEST_TIMEOUT}s），可能资源不足或 prompt 过长")
        return JSONResponse(
            status_code=504,
            content={"error": {"message": f"LLM inference timed out after {REQUEST_TIMEOUT}s", "type": "timeout_error"}},
        )
    except ValueError as e:
        print("[error] 上下文超限：", e)
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "invalid_request_error"}},
        )

    return response

# ---------------- 启动 ----------------
if __name__ == "__main__":
    print("[run] 服务已启动，监听 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
