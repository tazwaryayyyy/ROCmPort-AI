from openai import OpenAI
from groq import Groq
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

# pylint: disable=broad-exception-caught

# Load environment variables
load_dotenv()


class LLMClient:
    """
    Unified LLM client supporting three providers:
      1. Groq (default, local dev)         — GROQ_API_KEY
      2. vLLM on AMD Cloud (production)    — USE_VLLM=true + VLLM_* vars
      3. Qwen via HuggingFace Inference    — USE_QWEN=true + QWEN_API_KEY
         Model: Qwen/Qwen2.5-Coder-32B-Instruct (purpose-built for code tasks)
         Qualifies for the AMD hackathon Qwen bonus prize.
    """

    def __init__(self):
        self.use_vllm = os.getenv("USE_VLLM", "false").lower() == "true"
        self.use_qwen = os.getenv("USE_QWEN", "false").lower() == "true"
        self.client = None
        self.model = "mock"
        self.provider = "mock"
        self.init_error: Optional[str] = None

        if self.use_vllm:
            self._init_vllm()
        elif self.use_qwen:
            self._init_qwen()
        else:
            self._init_groq()

    # ------------------------------------------------------------------
    # Provider initializers
    # ------------------------------------------------------------------

    def _init_vllm(self) -> None:
        """Connect to vLLM endpoint on AMD Developer Cloud."""
        self.vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self.vllm_api_key = os.getenv("VLLM_API_KEY", "dummy-key")
        try:
            self.client = OpenAI(
                base_url=self.vllm_base_url,
                api_key=self.vllm_api_key
            )
            self.model = os.getenv("VLLM_MODEL", "amd/llama-3.3-70b")
            self.provider = "vLLM (AMD Cloud)"
        except Exception as e:
            self.init_error = f"vLLM client init failed: {str(e)}"
            print(f"Warning: {self.init_error}. Falling back to mock mode.")

    def _init_qwen(self) -> None:
        """
        Connect to Qwen/Qwen2.5-Coder-32B-Instruct via HuggingFace Inference API.

        Qwen2.5-Coder-32B-Instruct is purpose-built for code tasks and is directly
        relevant to CUDA-to-HIP translation. Free tier on HuggingFace — no billing.
        Set USE_QWEN=true and QWEN_API_KEY=hf_... in .env to activate.
        """
        qwen_api_key = os.getenv("QWEN_API_KEY")
        if not qwen_api_key:
            print("Warning: QWEN_API_KEY not found. Falling back to Groq.")
            self._init_groq()
            return
        try:
            # HuggingFace Inference API exposes an OpenAI-compatible endpoint
            hf_base_url = os.getenv(
                "QWEN_BASE_URL",
                "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1"
            )
            self.client = OpenAI(
                base_url=hf_base_url,
                api_key=qwen_api_key,
            )
            self.model = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
            self.provider = "Qwen (HuggingFace)"
        except Exception as e:
            self.init_error = f"Qwen client init failed: {str(e)}"
            print(f"Warning: {self.init_error}. Falling back to Groq.")
            self._init_groq()

    def _init_groq(self) -> None:
        """Connect to Groq (LLaMA-3.3-70B). Default provider for local development."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            print("Warning: GROQ_API_KEY not found. Using mock mode.")
            self.provider = "mock"
            return
        try:
            self.client = Groq(api_key=self.groq_api_key)
            self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            self.provider = "Groq (LLaMA-3.3-70B)"
        except Exception as e:
            self.init_error = f"Groq client init failed: {str(e)}"
            print(f"Warning: {self.init_error}. Falling back to mock mode.")
            self.provider = "mock"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def chat_completion(self, messages: list, temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """Send chat completion request to the configured LLM."""
        if self.client is None:
            # Mock response when no API key is available
            return (
                '{"kernels_found": ["mock_kernel"], "cuda_apis": ["cudaMalloc"], '
                '"warp_size_issue": true, "workload_type": "memory-bound", '
                '"sharding_detected": false, "difficulty": "Medium"}'
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        except Exception as e:
            message = str(e)
            lowered = message.lower()
            if "rate limit" in lowered or "429" in lowered or "quota" in lowered:
                raise RuntimeError(f"LLM request rate-limited: {message}") from e
            raise RuntimeError(f"LLM request failed: {message}") from e

    # ------------------------------------------------------------------
    # Utility / introspection
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        """Return current provider configuration for the /health and /benchmark-report endpoints."""
        return {
            "provider": self.provider,
            "model": self.model,
        }

    def test_connection(self) -> bool:
        """Test if the LLM connection is working."""
        try:
            test_messages = [
                {"role": "user", "content": "Respond with 'OK' if you can read this."}
            ]
            response = self.chat_completion(test_messages, max_tokens=10)
            return "OK" in response.upper()
        except Exception:
            return False
