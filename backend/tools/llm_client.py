import os
from typing import Optional, Dict, Any
from groq import Groq
from openai import OpenAI

class LLMClient:
    """Unified LLM client supporting both Groq (local) and vLLM (AMD Cloud)"""
    
    def __init__(self):
        self.use_vllm = os.getenv("USE_VLLM", "false").lower() == "true"
        
        if self.use_vllm:
            # vLLM configuration for AMD Cloud
            self.vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
            self.vllm_api_key = os.getenv("VLLM_API_KEY", "dummy-key")
            self.client = OpenAI(
                base_url=self.vllm_base_url,
                api_key=self.vllm_api_key
            )
            self.model = os.getenv("VLLM_MODEL", "amd/llama-3.3-70b")
        else:
            # Groq configuration for local development
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if not self.groq_api_key:
                print("Warning: GROQ_API_KEY not found. Using mock mode.")
                self.client = None
                self.model = "mock"
                return
            self.client = Groq(api_key=self.groq_api_key)
            self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    def chat_completion(self, messages: list, temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """Send chat completion request to the configured LLM"""
        if self.client is None:
            # Mock response when no API key is available
            return '{"kernels_found": ["mock_kernel"], "cuda_apis": ["cudaMalloc"], "warp_size_issue": true, "workload_type": "memory-bound", "sharding_detected": false, "difficulty": "Medium"}'
        
        try:
            if self.use_vllm:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
                
        except Exception as e:
            raise Exception(f"LLM request failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        if self.use_vllm:
            return {
                'provider': 'vLLM',
                'model': self.model,
                'base_url': self.vllm_base_url,
                'platform': 'AMD Cloud'
            }
        else:
            return {
                'provider': 'Groq',
                'model': self.model,
                'platform': 'Local Development'
            }
    
    def test_connection(self) -> bool:
        """Test if the LLM connection is working"""
        try:
            test_messages = [
                {"role": "user", "content": "Respond with 'OK' if you can read this."}
            ]
            response = self.chat_completion(test_messages, max_tokens=10)
            return "OK" in response.upper()
        except:
            return False
