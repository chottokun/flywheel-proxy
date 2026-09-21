from typing import Any


def clean_request_for_vendor(route: str, payload: dict[str, Any]) -> dict[str, Any]:
    """各ベンダーの互換エンドポイントでエラーを引き起こす独自パラメータを除去・正規化する"""
    cleaned = dict(payload)

    if route == "route_a":
        # vLLM 向け: stream_options 等の対応状況に応じて維持
        pass
    elif route == "route_b":
        # Gemini OpenAI 互換エンドポイント向けサニタイズ
        # stream_options は未サポートの場合があるため除去
        cleaned.pop("stream_options", None)
    elif route == "route_c":
        # OpenAI / Anthropic 向け
        pass

    return cleaned
