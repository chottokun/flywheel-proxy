from app.sanitizer import clean_request_for_vendor
from app.schemas import ChatCompletionRequest, ChatMessage, ModelCard, ModelListResponse


def test_chat_message_extra():
    msg = ChatMessage(role="user", content="hello", name="test_user")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.model_extra == {"name": "test_user"}

def test_chat_completion_request_extra():
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stream_options={"include_usage": True},
        temperature=0.7
    )
    assert req.model == "test-model"
    assert req.stream is False
    assert req.model_extra == {"stream_options": {"include_usage": True}, "temperature": 0.7}

def test_model_list_schemas():
    card1 = ModelCard(id="auto")
    card2 = ModelCard(id="route_a")
    res = ModelListResponse(data=[card1, card2])
    assert len(res.data) == 2
    assert res.data[0].id == "auto"

def test_clean_request_for_vendor():
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "stream_options": {"include_usage": True},
        "temperature": 0.5
    }

    # route_a keeps all
    cleaned_a = clean_request_for_vendor("route_a", payload)
    assert "stream_options" in cleaned_a
    assert cleaned_a["temperature"] == 0.5

    # route_b (Gemini) removes stream_options
    cleaned_b = clean_request_for_vendor("route_b", payload)
    assert "stream_options" not in cleaned_b
    assert cleaned_b["temperature"] == 0.5

    # route_c keeps all
    cleaned_c = clean_request_for_vendor("route_c", payload)
    assert "stream_options" in cleaned_c
    assert cleaned_c["temperature"] == 0.5
