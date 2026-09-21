import abc
from typing import Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: Any = ""
    name: Any = None
    tool_calls: Any = None
    tool_call_id: Any = None


class BaseRouterEngine(abc.ABC):
    @abc.abstractmethod
    async def determine_route(self, messages: list[ChatMessage]) -> tuple[str, dict[str, Any]]:
        """
        メッセージ履歴からルーティング先とメトリクス辞書を返す。
        Returns:
            (route, metrics_dict)
            route: 'route_a', 'route_b', 'route_c' のいずれか
        """
