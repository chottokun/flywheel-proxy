import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.config import settings
from app.engine.base import BaseRouterEngine
from app.logger import log_event, setup_logging, shutdown_logging

logger = logging.getLogger("flywheel.main")

http_client: httpx.AsyncClient | None = None
router_engine: BaseRouterEngine | None = None
transaction_storage = None  # Jules 実装の storage.py が pull された後に接続


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, router_engine, transaction_storage
    setup_logging()

    limits = httpx.Limits(
        max_keepalive_connections=settings.HTTP_MAX_KEEP_ALIVE_CONNECTIONS,
        max_connections=settings.HTTP_MAX_CONNECTIONS,
    )
    http_client = httpx.AsyncClient(
        limits=limits,
        timeout=httpx.Timeout(
            settings.HTTP_TIMEOUT_SECONDS,
            connect=settings.HTTP_CONNECT_TIMEOUT_SECONDS,
        ),
    )

    # ルーターエンジンの初期化
    if settings.ROUTER_TYPE == "logit_router":
        from app.engine.logit_router import LogitRouterEngine

        router_engine = LogitRouterEngine()
    else:
        logger.warning(f"Unknown ROUTER_TYPE: {settings.ROUTER_TYPE}")

    # Storage の初期化（モジュールが存在する場合）
    try:
        from app.storage import TransactionStorage

        transaction_storage = TransactionStorage(settings.DB_PATH)
        await transaction_storage.initialize()
    except ImportError:
        pass

    yield

    if transaction_storage:
        await transaction_storage.close()
    await http_client.aclose()
    shutdown_logging()


app = FastAPI(
    title="Flywheel Pro LLM Proxy",
    description="Logit-Router integrated self-evolving LLM pass-through proxy",
    version="0.1.0",
    lifespan=lifespan,
)


def resolve_backend_config(route: str):
    if route == "route_a":
        return settings.LOCAL_LLM_URL, settings.LOCAL_LLM_MODEL, settings.LOCAL_LLM_API_KEY
    elif route == "route_b":
        return settings.COMMERCIAL_FAST_URL, settings.COMMERCIAL_FAST_MODEL, settings.GEMINI_API_KEY
    else:
        return settings.COMMERCIAL_EXPERT_URL, settings.COMMERCIAL_EXPERT_MODEL, settings.OPENAI_API_KEY


async def process_collected_chunks(
    trace_id: str,
    route: str,
    finish_reason: str,
    raw_chunks: list[bytes],
    metrics: dict,
    messages_payload: list,
):
    """ストリーム完了後に別タスクで非同期実行されるテキスト復元 & 永続化ワーカー"""
    full_text = ""
    try:
        combined = b"".join(raw_chunks).decode("utf-8", errors="ignore")
        for line in combined.split("\n"):
            line = line.strip()
            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                data = json.loads(line[6:])
                delta = data.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    full_text += delta["content"]
    except Exception:
        pass

    log_event(
        "proxy.stream",
        "stream_completed",
        {
            "trace_id": trace_id,
            "route": route,
            "finish_reason": finish_reason,
            "response_text_length": len(full_text),
        },
    )

    if transaction_storage:
        record = {
            "trace_id": trace_id,
            "route": route,
            "latency_ms": metrics.get("latency_ms", 0.0),
            "entropy": metrics.get("entropy", 0.0),
            "margin": metrics.get("margin", 0.0),
            "escalated": metrics.get("escalated", False),
            "finish_reason": finish_reason,
            "messages": messages_payload,
            "response_text": full_text,
        }
        await transaction_storage.log_transaction(record)


async def stream_and_tee(
    upstream_resp: httpx.Response,
    client_request: Request,
    trace_id: str,
    route: str,
    metrics: dict,
    messages_payload: list,
) -> AsyncGenerator[bytes, None]:
    raw_chunks = []
    finish_reason = "completed"

    try:
        async for chunk in upstream_resp.aiter_bytes():
            # クライアント切断を能動検知し上流を強制切断
            if await client_request.is_disconnected():
                finish_reason = "client_abort"
                await upstream_resp.aclose()
                break

            yield chunk
            raw_chunks.append(chunk)

    except Exception as e:
        finish_reason = f"error: {e!s}"
        raise
    finally:
        await upstream_resp.aclose()
        # テキスト復元とDB保存はイベントループを塞がずバックグラウンドタスクとして発火
        asyncio.create_task(
            process_collected_chunks(
                trace_id, route, finish_reason, raw_chunks, metrics, messages_payload
            )
        )


@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    trace_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        body = await raw_request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages_data = body.get("messages", [])

    # ルーティング判定
    from app.engine.base import ChatMessage

    chat_messages = [ChatMessage(**m) for m in messages_data]
    route, metrics = await router_engine.determine_route(chat_messages)
    decision_ms = (time.perf_counter() - start_time) * 1000
    metrics["latency_ms"] = decision_ms

    log_event(
        "proxy.router",
        "router_decision",
        {"trace_id": trace_id, "latency_ms": decision_ms, **metrics},
    )

    # 上流バックエンドの設定
    url, model_name, api_key = resolve_backend_config(route)
    payload = dict(body)
    payload["model"] = model_name

    # サニタイザー適用（モジュールが存在する場合）
    try:
        from app.sanitizer import clean_request_for_vendor

        payload = clean_request_for_vendor(route, payload)
    except ImportError:
        pass

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async def send_upstream(target_url, target_payload, target_headers):
        req_upstream = http_client.build_request(
            "POST",
            f"{target_url}/chat/completions",
            json=target_payload,
            headers=target_headers,
        )
        return await http_client.send(req_upstream, stream=body.get("stream", False))

    try:
        upstream_resp = await send_upstream(url, payload, headers)
        if upstream_resp.status_code >= 500:
            raise httpx.HTTPStatusError(
                "Server error", request=upstream_resp.request, response=upstream_resp
            )
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError):
        # 自動フェイルオーバー: 最上位 Route C へ再試行
        url, model_name, api_key = resolve_backend_config("route_c")
        payload["model"] = model_name
        try:
            from app.sanitizer import clean_request_for_vendor

            payload = clean_request_for_vendor("route_c", payload)
        except ImportError:
            pass
        headers["Authorization"] = f"Bearer {api_key}"
        upstream_resp = await send_upstream(url, payload, headers)

    if body.get("stream", False):
        return StreamingResponse(
            stream_and_tee(upstream_resp, raw_request, trace_id, route, metrics, messages_data),
            media_type="text/event-stream",
        )
    else:
        resp_bytes = await upstream_resp.aread()
        return Response(
            content=resp_bytes,
            status_code=upstream_resp.status_code,
            media_type="application/json",
        )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "auto", "object": "model", "owned_by": "flywheel-proxy"},
            {"id": "route_a", "object": "model", "owned_by": "flywheel-proxy"},
            {"id": "route_b", "object": "model", "owned_by": "flywheel-proxy"},
            {"id": "route_c", "object": "model", "owned_by": "flywheel-proxy"},
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "router_type": settings.ROUTER_TYPE,
        "router_model": settings.ROUTER_MODEL_ID,
    }
