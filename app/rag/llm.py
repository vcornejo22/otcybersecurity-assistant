"""Nan Builders chat model wrapper for LangChain."""

import json
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import Settings
from app.rag.exceptions import LLMUnavailableError


class NanBuildersChatModel(BaseChatModel):
    """LangChain-compatible chat model for the Nan Builders API.

    Calls ``POST /v1/chat/completions`` with configurable model and
    temperature. The API key and base URL are read from the application
    settings.
    """

    settings: Settings | None = None
    model: str | None = None
    temperature: float = 0.3
    max_tokens: int | None = None
    timeout: float = 60.0

    def __init__(self, settings: Settings | None = None, **kwargs: Any) -> None:
        super().__init__(settings=settings, **kwargs)
        self.settings = settings or Settings()
        if not self.settings.LLM_API_KEY:
            raise LLMUnavailableError(
                "LLM_API_KEY is not configured. Set it in your environment or .env file."
            )
        if self.model is None:
            self.model = self.settings.LLM_MODEL
        if self.max_tokens is None:
            self.max_tokens = self.settings.LLM_MAX_TOKENS
        self._client = httpx.Client(
            base_url=self.settings.LLM_BASE_URL,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.settings.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
        )

    @property
    def _llm_type(self) -> str:
        return "nan-builders"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.settings.LLM_BASE_URL,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        api_messages = [_convert_message(m) for m in messages]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop

        try:
            response = self._client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise LLMUnavailableError("Nan Builders LLM API is unreachable or timed out") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise LLMUnavailableError(
                    f"Nan Builders LLM API returned {exc.response.status_code}"
                ) from exc
            raise

        try:
            payload_data = response.json()
            choice = payload_data["choices"][0]
            message_data = choice["message"]
            content = message_data.get("content")
            finish_reason = choice.get("finish_reason", "unknown")

            # Handle null/empty content
            if content is None:
                refusal = message_data.get("refusal")
                if refusal:
                    raise LLMUnavailableError(f"El modelo rechazó responder: {refusal}")
                # Some APIs put content in tool_calls
                tool_calls = message_data.get("tool_calls")
                if tool_calls:
                    raise LLMUnavailableError(
                        "El modelo devolvió tool_calls en vez de contenido de texto. "
                        "Posiblemente el modelo no está configurado para chat directo."
                    )
                raise LLMUnavailableError(
                    f"El modelo devolvió contenido vacío (finish_reason={finish_reason}). "
                    f"Response: {json.dumps(payload_data, default=str)[:500]}"
                )

            usage = payload_data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMUnavailableError(f"Malformed LLM response: {response.text[:500]}") from exc

        message = AIMessage(
            content=content,
            response_metadata={
                "model": self.model,
                "tokens_used": tokens_used,
            },
        )
        generation = ChatGeneration(message=message, text=content)
        return ChatResult(generations=[generation], llm_output=usage)


def _convert_message(message: BaseMessage) -> dict[str, str]:
    """Map a LangChain message to the Nan Builders API message format."""
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "user"
    return {"role": role, "content": str(message.content)}
