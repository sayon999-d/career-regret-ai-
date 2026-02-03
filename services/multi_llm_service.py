from typing import Dict, List, Optional, Any
from enum import Enum
import abc

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"

class LLMInterface(abc.ABC):
    @abc.abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

class MockLLM(LLMInterface):
    async def generate(self, prompt: str, **kwargs) -> str:
        return f"[Mock Response for: {prompt[:50]}...]"

class MultiLLMService:
    def __init__(self):
        self.providers: Dict[LLMProvider, Any] = {
            LLMProvider.MOCK: MockLLM()
        }
        self.active_provider = LLMProvider.MOCK
        self.model_configs: Dict[str, Any] = {
            "default": {"temp": 0.7, "max_tokens": 1000}
        }

    def register_provider(self, name: LLMProvider, provider_instance: Any):
        self.providers[name] = provider_instance

    def set_active_provider(self, name: LLMProvider):
        if name in self.providers:
            self.active_provider = name
            return True
        return False

    async def get_response(self, prompt: str, provider: Optional[LLMProvider] = None, **kwargs) -> str:
        target_provider = provider or self.active_provider
        handler = self.providers.get(target_provider, self.providers[LLMProvider.MOCK])

        config = {**self.model_configs["default"], **kwargs}

        try:
            return await handler.generate(prompt, **config)
        except Exception as e:
            print(f"LLM Error ({target_provider}): {e}")
            if target_provider != LLMProvider.MOCK:
                return await self.providers[LLMProvider.MOCK].generate(prompt)
            return "Internal LLM Error"

multi_llm_service = MultiLLMService()
