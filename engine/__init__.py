"""
Intelligent Email Assistant Engine Package
Course: 24CS2506 – Building Applications using GPT-4
Experiment 3: Intelligent Email Assistant using Ollama
"""

from .ollama_client import OllamaClient
from .intent_analyzer import EmailAnalyzer
from .memory_retriever import InteractionRetriever
from .analytics import EmailAnalytics
from .email_generator import IntelligentEmailAssistant

__all__ = [
    "OllamaClient",
    "EmailAnalyzer",
    "InteractionRetriever",
    "EmailAnalytics",
    "IntelligentEmailAssistant"
]
