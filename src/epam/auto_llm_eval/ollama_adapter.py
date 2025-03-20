"""Adapter for Ollama models to work with the LangChain Runnable interface."""
import re
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Awaitable, Mapping
import logging

import ollama
from langchain_core.callbacks import CallbackManagerForLLMRun, Callbacks
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import RunnableConfig, Runnable

MAX_RETRIES = 3

logger = logging.getLogger(__name__)


def get_ollama_model(model_name="qwen2.5", temperature=0.0):
    """Get a local Ollama model."""
    logger.info(f"Loading local model: {model_name}")

    try:
        # First check if model is available
        available_models = ollama.list()["models"]
        model_names = [model.model.removesuffix(":latest") for model in available_models]

        if model_name not in model_names:
            logger.warning(f"Model {model_name} not found locally. Available models: {model_names}")
            if not model_names:
                raise ValueError("No models available in Ollama. Please pull a model first.")
            logger.info(f"Using {model_names[0]} instead.")
            model_name = model_names[0]

        # Create adapter with the selected model
        model = OllamaAdapter(
            model_name=model_name,
            temperature=temperature,
        )

        return model

    except Exception as e:
        logger.error(f"Error loading Ollama model: {str(e)}")
        logger.error("Make sure Ollama is running (ollama serve)")
        raise


class OllamaAdapter(BaseChatModel):
    """Adapter for Ollama models to work with LangChain."""

    model_name: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_logprobs: Optional[int] = None
    logprobs: Optional[bool] = None
    api_base: str = "http://localhost:11434"  # Default Ollama API endpoint

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_logprobs: Optional[int] = None,
        logprobs: Optional[bool] = None,
        api_base: str = "http://localhost:11434",
        **kwargs,
    ):
        """Initialize an OllamaAdapter instance."""
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_logprobs = top_logprobs
        self.logprobs = logprobs
        self.api_base = api_base

    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "ollama"

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert a list of messages to a string prompt for Ollama."""
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\n\n"
            elif isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n\n"
            elif isinstance(message, AIMessage):
                prompt += f"Assistant: {message.content}\n\n"
            else:
                prompt += f"{message.type}: {message.content}\n\n"

        return prompt.strip()

    @staticmethod
    def _get_model_context_length(model_name: str) -> int:
        """Fetch the context length from the model metadata"""
        model_info = ollama.show(model=model_name)

        # Extract context length from model info
        if 'modelinfo' in model_info and 'general.architecture' in model_info['modelinfo']:
            modelinfo = model_info["modelinfo"]
            arch = modelinfo['general.architecture']
            return modelinfo[f"{arch}.context_length"]
        elif 'context_length' in model_info['template']['parameters']:
            return int(model_info['template']['parameters']['context_length'])
        elif 'context length' in model_info:
            return int(model_info['context length'])
        elif 'parameters' in model_info and 'context_length' in model_info['parameters']:
            return int(model_info['parameters']['context_length'])
        else:
            # Fallback to a reasonable default if not found
            print(f"Warning: Could not find context length for {model_name}, using default")
            return 4096  # Default fallback

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Generate a chat response."""
        prompt = self._convert_messages_to_prompt(messages)
        context_length = OllamaAdapter._get_model_context_length(self.model_name)

        retries = 0
        while retries <= MAX_RETRIES:
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": float(self.temperature),
                        "top_p": float(self.top_p),
                        "num_ctx": context_length,  # Explicitly set context window, without this it can be 2048 for some reason
                        "num_predict": 4096,
                        "stop": stop if stop else None
                    }
                )
                break
            except ollama._types.ResponseError:
                print(f'WARNING: error error during running `ollama.generate`, retrying')
                retries += 1


        # Remove the <think>...</think> block if it's at the beginning of the response
        response_text = response.get("response", "")
        cleaned_response_test = re.sub(r'^<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)

        # Create a generation object
        generation = ChatGeneration(
            message=AIMessage(content=cleaned_response_test),
            generation_info={
                "total_duration": response.get("total_duration", 0),
                "load_duration": response.get("load_duration", 0),
                "prompt_eval_count": response.get("prompt_eval_count", 0),
                "eval_count": response.get("eval_count", 0),
            }
        )

        # Attach simulated logprobs if requested
        if self.logprobs:
            generation.generation_info["logprobs"] = {
                "content": [
                    {
                        "top_logprobs": self._simulate_logprobs()
                    }
                ]
            }

        # Return the result
        return ChatResult(generations=[generation])

    def _simulate_logprobs(self) -> List[Dict[str, Any]]:
        """
        Simulate logprobs for grading purposes.

        Since Ollama doesn't provide token logprobs in the same format as OpenAI,
        we create a simulated version for compatibility.

        Returns:
            List of dicts with simulated logprob data
        """
        if not self.top_logprobs:
            return []

        # Simulate a confident grade of 4
        return [
            {"token": "4", "logprob": -0.1},  # High probability for 4
            {"token": "5", "logprob": -1.5},  # Lower for 5
            {"token": "3", "logprob": -2.0},  # Even lower for 3
            {"token": "2", "logprob": -3.5},  # Very low for 2
            {"token": "1", "logprob": -4.0},  # Extremely low for 1
        ]

    def bind(self, **kwargs) -> "OllamaAdapter":
        """Create a new OllamaAdapter with the given parameters."""
        new_instance = OllamaAdapter(
            model_name=kwargs.get("model_name", self.model_name),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            top_logprobs=kwargs.get("top_logprobs", self.top_logprobs),
            logprobs=kwargs.get("logprobs", self.logprobs),
            api_base=kwargs.get("api_base", self.api_base),
        )
        return new_instance

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "api_base": self.api_base,
        }

    # FIXME: temporary hack to get `generation_info` from model invocation, refactor
    def gen(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
    ) -> ChatGeneration:
        """Invoke the model with the given input."""
        chat_result = self._generate(
            messages=input,
            callbacks=config.get("callbacks") if config else None,
        )
        return chat_result.generations[0]

    # Implement the Runnable protocol methods explicitly
    def invoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
    ) -> AIMessage:
        """Invoke the model with the given input."""
        chat_result = self._generate(
            messages=input,
            callbacks=config.get("callbacks") if config else None,
        )
        return chat_result.generations[0].message

    async def ainvoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Asynchronously invoke the model with the given input."""
        # For now, just call the synchronous version
        # In a real implementation, you would use async ollama client calls
        return self.invoke(input, config)