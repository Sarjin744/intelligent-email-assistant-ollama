"""
Ollama Client Module for Experiment 3
Course: 24CS2506 – Building Applications using GPT-4
Experiment 3: Intelligent Email Assistant using Ollama
"""

import json
import requests
from typing import Dict, Any, Optional, List, Generator


class OllamaClient:
    """
    Client for interacting with local Ollama API (http://localhost:11434).
    Supports generation, model querying, and streaming responses for Llama 3.2.
    """
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._last_conn_check = 0.0
        self._conn_cached = False

    def check_connection(self, force: bool = False) -> bool:
        """Check if local Ollama daemon is reachable with short cache."""
        import time
        now = time.time()
        if not force and (now - self._last_conn_check < 4.0):
            return self._conn_cached

        self._last_conn_check = now
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=0.8)
            self._conn_cached = (resp.status_code == 200)
            return self._conn_cached
        except Exception:
            self._conn_cached = False
            return False

    def list_models(self) -> List[str]:
        """List available local models in Ollama."""
        if not self.check_connection():
            return []
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return [m for m in models if m]
            return []
        except Exception:
            return []

    def get_best_available_model(self) -> str:
        """Return llama3.2 if available, or first available model, or default."""
        models = self.list_models()
        if not models:
            return self.default_model
        for m in models:
            if "llama3.2" in m.lower():
                return m
        for m in models:
            if any(k in m.lower() for k in ["llama", "qwen", "mistral", "phi", "gemma"]):
                return m
        return models[0]

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """Generate response using Ollama /api/generate endpoint."""
        if not self.check_connection():
            return None

        target_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if system:
            payload["system"] = system
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120.0
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "").strip()
        except Exception:
            return None
        return None

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.5
    ) -> Generator[str, None, None]:
        """Stream generation response chunks from Ollama."""
        if not self.check_connection():
            return

        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature}
        }
        if system:
            payload["system"] = system

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120.0
            ) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line.decode("utf-8"))
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
        except Exception:
            return
