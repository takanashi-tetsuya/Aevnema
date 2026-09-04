from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from google import genai
import httpx
from dotenv import load_dotenv
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _consume_task_result(task: asyncio.Task) -> None:
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


def _trace_enabled() -> bool:
    return os.getenv("ENABLE_TRACE_LOGGING", "false").casefold() == "true"


def _trace_content_enabled() -> bool:
    return os.getenv("ENABLE_TRACE_CONTENT_LOGGING", "false").casefold() == "true"


def _content_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(value))


class Message(BaseModel):
    role: str # "user", "assistant", or "system"
    content: Union[str, List[Dict[str, Any]]]

class GenerationOptions(BaseModel):
    """Validated per-call overrides; unset values inherit model configuration."""

    model_config = ConfigDict(extra="forbid")

    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=131_072)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = Field(default=None, ge=1)
    response_format: Optional[Dict[str, Any]] = None
    extra_body: Dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    provider: str # "openai" or "gemini"
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=131_072)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = Field(default=None, ge=1)
    response_format: Optional[Dict[str, Any]] = None
    extra_body: Dict[str, Any] = Field(default_factory=dict)

    def with_options(
        self, options: GenerationOptions | Dict[str, Any] | None
    ) -> "LLMConfig":
        if options is None:
            return self
        parsed = (
            options
            if isinstance(options, GenerationOptions)
            else GenerationOptions.model_validate(options)
        )
        updates = parsed.model_dump(exclude_unset=True)
        option_extra = updates.pop("extra_body", None)
        if option_extra:
            updates["extra_body"] = {**self.extra_body, **option_extra}
        return self.model_copy(update=updates)

class LLMEngine:
    def __init__(
        self,
        fallback_configs: List[LLMConfig],
        *,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        request_deadline_seconds: float | None = None,
    ):
        """
        初始化 LLM 引擎
        :param fallback_configs: 一个按优先级排序的模型配置列表。
                                 例如: [Config(Gemini), Config(OpenAI)]
        """
        if not fallback_configs:
            raise ValueError("至少需要提供一个 LLM 配置")
        self.configs = fallback_configs
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.request_deadline_seconds = (
            None
            if request_deadline_seconds is None
            else max(0.1, float(request_deadline_seconds))
        )

    def describe(self) -> dict[str, Any]:
        """Return a secret-free description suitable for status commands."""

        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "request_deadline_seconds": self.request_deadline_seconds,
            "models": [
                {
                    key: value
                    for key, value in {
                        "provider": item.provider,
                        "model": item.model_name,
                        "base_url": item.base_url,
                        "timeout_seconds": item.timeout_seconds,
                        "temperature": item.temperature,
                        "max_tokens": item.max_tokens,
                        "top_p": item.top_p,
                        "top_k": item.top_k,
                        "seed": item.seed,
                        "presence_penalty": item.presence_penalty,
                        "frequency_penalty": item.frequency_penalty,
                        "stop": item.stop,
                        "enable_thinking": item.enable_thinking,
                        "thinking_budget": item.thinking_budget,
                        "extra_body": item.extra_body or None,
                    }.items()
                    if value is not None
                }
                for item in self.configs
            ],
        }

    def _call_openai(self, config: LLMConfig, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        oai_messages = []
        if system_prompt:
            oai_messages.append({"role": "system", "content": system_prompt})
            
        for msg in messages:
            # 转换成 openai 标准格式
            oai_messages.append({"role": msg.role, "content": msg.content})

        if _trace_enabled():
            logger.info(
                "[TRACE] OpenAI request model=%s messages=%d input_chars=%d",
                config.model_name,
                len(oai_messages),
                sum(_content_chars(item.get("content")) for item in oai_messages),
            )
            if _trace_content_enabled():
                logger.info("[TRACE-CONTENT] Messages: %s", oai_messages)

        extra_body: dict[str, Any] = dict(config.extra_body)
        if config.enable_thinking is not None:
            extra_body["enable_thinking"] = config.enable_thinking
        if config.thinking_budget is not None and config.enable_thinking is not False:
            extra_body["thinking_budget"] = config.thinking_budget
        if config.top_k is not None:
            extra_body["top_k"] = config.top_k
        request: dict[str, Any] = {
            "model": config.model_name,
            "messages": oai_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        for key in (
            "top_p",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "response_format",
        ):
            value = getattr(config, key)
            if value is not None:
                request[key] = value
        if extra_body:
            request.update(extra_body)
        base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json=request,
            timeout=httpx.Timeout(config.timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
        reply_content = payload["choices"][0]["message"].get("content")
        if _trace_enabled():
            logger.info(
                "[TRACE] OpenAI response model=%s output_chars=%d",
                config.model_name,
                _content_chars(reply_content),
            )
            if _trace_content_enabled():
                logger.info("[TRACE-CONTENT] OpenAI response: %s", reply_content)
            
        return reply_content or ""

    async def _call_gemini(self, config: LLMConfig, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        client = genai.Client(api_key=config.api_key)
        
        # Gemini 的模型配置
        gemini_config = {
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
        }
        if config.top_p is not None:
            gemini_config["top_p"] = config.top_p
        if config.top_k is not None:
            gemini_config["top_k"] = config.top_k
        if config.stop is not None:
            gemini_config["stop_sequences"] = (
                [config.stop] if isinstance(config.stop, str) else config.stop
            )
        if config.seed is not None:
            gemini_config["seed"] = config.seed
        if system_prompt:
             gemini_config["system_instruction"] = system_prompt

        # 转换成 gemini 标准格式 (user / model)
        gemini_messages = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            gemini_messages.append({"role": role, "parts": [{"text": msg.content}]})

        if _trace_enabled():
            logger.info(
                "[TRACE] Gemini request model=%s messages=%d input_chars=%d system_chars=%d",
                config.model_name,
                len(gemini_messages),
                sum(_content_chars(item) for item in gemini_messages),
                _content_chars(system_prompt),
            )
            if _trace_content_enabled():
                logger.info("[TRACE-CONTENT] System: %s", system_prompt)
                logger.info("[TRACE-CONTENT] Messages: %s", gemini_messages)

        # genai SDK currently uses generate_content for both sync and async via client.aio
        response = await client.aio.models.generate_content(
            model=config.model_name,
            contents=gemini_messages,
            config=genai.types.GenerateContentConfig(**gemini_config)
        )
        
        reply_content = response.text
        if _trace_enabled():
            logger.info(
                "[TRACE] Gemini response model=%s output_chars=%d",
                config.model_name,
                _content_chars(reply_content),
            )
            if _trace_content_enabled():
                logger.info("[TRACE-CONTENT] Gemini response: %s", reply_content)
            
        return reply_content or ""

    async def generate_response(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        task_context: str = "",
        request_options: GenerationOptions | Dict[str, Any] | None = None,
        deadline_seconds: float | None = None,
    ) -> str:
        """
        尝试按优先级调用 LLM，如果失败会重试 n 次，如果均失败则自动回退(Fallback)到下一个配置
        """
        attempts = self.max_retries if max_retries is None else max(1, int(max_retries))
        delay = self.retry_delay if retry_delay is None else max(0.0, float(retry_delay))
        effective_deadline = (
            self.request_deadline_seconds
            if deadline_seconds is None
            else max(0.1, float(deadline_seconds))
        )
        loop = asyncio.get_running_loop()
        deadline_at = (
            loop.time() + effective_deadline
            if effective_deadline is not None
            else None
        )
        errors = []
        prefix = f"[{task_context}] " if task_context else ""
        for i, base_config in enumerate(self.configs):
            config = base_config.with_options(request_options)
            for attempt in range(attempts):
                try:
                    remaining = (
                        None if deadline_at is None else deadline_at - loop.time()
                    )
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError(
                            f"task deadline exceeded ({effective_deadline:.1f}s)"
                        )
                    logger.info(f"{prefix}正在尝试使用优先级 {i+1} 的模型: {config.provider} ({config.model_name}) (第 {attempt+1} 次尝试)")
                    if config.provider.lower() == "openai":
                        call_config = config.model_copy(
                            update={
                                "timeout_seconds": min(
                                    float(config.timeout_seconds),
                                    max(0.1, float(remaining))
                                    if remaining is not None
                                    else float(config.timeout_seconds),
                                )
                            }
                        )
                        operation = asyncio.to_thread(
                            self._call_openai,
                            call_config,
                            messages,
                            system_prompt,
                        )
                    elif config.provider.lower() == "gemini":
                        operation = self._call_gemini(
                            config, messages, system_prompt
                        )
                    else:
                        raise ValueError(f"不支持的提供商: {config.provider}")
                    if remaining is None:
                        res = await operation
                    else:
                        operation_task = asyncio.create_task(operation)
                        completed, _pending = await asyncio.wait(
                            {operation_task}, timeout=remaining
                        )
                        if not completed:
                            operation_task.cancel()
                            operation_task.add_done_callback(
                                _consume_task_result
                            )
                            raise TimeoutError(
                                f"task deadline exceeded "
                                f"({effective_deadline:.1f}s)"
                            )
                        res = operation_task.result()
                        
                    if not res or not res.strip():
                        raise ValueError(f"模型 {config.model_name} 返回了空内容，可能被过滤或服务异常")
                    return res
                except Exception as e:
                    logger.warning(f"{prefix}模型 {config.provider} ({config.model_name}) 第 {attempt+1} 次调用失败: {e}")
                    if attempt == attempts - 1:
                        errors.append(
                            f"{config.provider}/{config.model_name}: "
                            f"{type(e).__name__}: {e}"
                        )
                    else:
                        if delay:
                            await asyncio.sleep(delay)
                    continue
                
        raise RuntimeError(f"所有备用模型均调用失败。错误信息: {errors}")


class EngineFactory:
    @staticmethod
    def _merge_config(*layers: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            for key, value in layer.items():
                if key == "extra_body" and isinstance(value, dict):
                    extra_body.update(value)
                else:
                    merged[key] = value
        if extra_body:
            merged["extra_body"] = extra_body
        return merged

    @staticmethod
    def create(task_type: str = "chat") -> LLMEngine:
        import tomllib

        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
        configured_path = Path(
            os.getenv("MODEL_CONFIG_PATH", "config/model_config.toml")
        )
        config_path = (
            configured_path
            if configured_path.is_absolute()
            else project_root / configured_path
        )
        with config_path.open("rb") as handle:
            config_data = tomllib.load(handle)

        engines_config = config_data.get("engines", {})
        selected_task = task_type if task_type in engines_config else "chat"
        models_value = engines_config.get(selected_task, [])
        models_list = models_value if isinstance(models_value, list) else [models_value]
        if not models_list:
            raise ValueError(f"配置中未找到 {selected_task} 的可用模型。")

        defaults = config_data.get("defaults", {})
        task_defaults = config_data.get("task_defaults", {}).get(selected_task, {})
        fallback_configs: list[LLMConfig] = []
        for model in models_list:
            merged = EngineFactory._merge_config(defaults, task_defaults, model)
            key_env = str(merged.get("api_key_env", "SILICONFLOW_API_KEY"))
            api_key = str(merged.get("api_key") or os.getenv(key_env, ""))
            if not api_key:
                raise ValueError(
                    f"模型 {merged.get('model', '<unknown>')} 缺少 API key；"
                    f"请设置 {key_env} 或 api_key"
                )
            fallback_configs.append(
                LLMConfig(
                    provider=str(merged.get("provider", "openai")),
                    model_name=str(merged.get("model", "")),
                    api_key=api_key,
                    base_url=merged.get("base_url"),
                    timeout_seconds=float(merged.get("timeout_seconds", 300.0)),
                    temperature=float(merged.get("temperature", 0.7)),
                    max_tokens=int(merged.get("max_tokens", 1000)),
                    top_p=merged.get("top_p"),
                    top_k=merged.get("top_k"),
                    seed=merged.get("seed"),
                    presence_penalty=merged.get("presence_penalty"),
                    frequency_penalty=merged.get("frequency_penalty"),
                    stop=merged.get("stop"),
                    enable_thinking=merged.get("enable_thinking"),
                    thinking_budget=merged.get("thinking_budget"),
                    response_format=merged.get("response_format"),
                    extra_body=merged.get("extra_body", {}),
                )
            )

        runtime = config_data.get("runtime", {})
        return LLMEngine(
            fallback_configs=fallback_configs,
            max_retries=int(runtime.get("max_retries", 3)),
            retry_delay=float(runtime.get("retry_delay", 1.0)),
            request_deadline_seconds=(
                float(task_defaults["deadline_seconds"])
                if task_defaults.get("deadline_seconds") is not None
                else None
            ),
        )
