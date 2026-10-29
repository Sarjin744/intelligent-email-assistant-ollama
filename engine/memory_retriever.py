"""
Previous Interactions Memory Retriever
Course: 24CS2506 – Building Applications using GPT-4
Experiment 3: Intelligent Email Assistant using Ollama

Implements semantic matching of customer emails with historical interactions
using TF-IDF Vectorization and Cosine Similarity.
"""

import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class InteractionRetriever:
    """
    Retrieves the most semantically relevant historical email interactions
    to provide high-quality Few-Shot context to the LLM.
    """
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "past_interactions.json"
        )
        self.interactions: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self.load_interactions()

    def load_interactions(self):
        """Load interaction records from JSON storage and build TF-IDF matrix."""
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.interactions = json.load(f)
        else:
            self.interactions = self._default_fallback_dataset()

        # Build corpus from past emails + category keywords
        corpus = [
            f"{item['customer_email']} {' '.join(item.get('keywords', []))}"
            for item in self.interactions
        ]
        if corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def find_similar(self, query_email: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Compute Cosine Similarity between query email and past interactions.
        Returns the top-k most similar previous interactions with similarity scores.
        """
        if not self.interactions or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query_email])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Rank indices by descending similarity
        ranked_indices = np.argsort(sim_scores)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            score = float(sim_scores[idx])
            item = self.interactions[idx].copy()
            item["similarity_score"] = round(score, 4)
            item["similarity_percentage"] = round(score * 100, 1)
            results.append(item)

        return results

    def add_interaction(self, customer_email: str, agent_reply: str, category: str = "General", keywords: List[str] = None):
        """Append a newly approved interaction to the knowledge base and retrain."""
        new_entry = {
            "id": f"PI-{len(self.interactions) + 101}",
            "category": category,
            "customer_email": customer_email.strip(),
            "agent_reply": agent_reply.strip(),
            "keywords": keywords or []
        }
        self.interactions.append(new_entry)

        # Re-save to JSON
        try:
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self.interactions, f, indent=2)
        except Exception:
            pass

        # Re-index
        corpus = [
            f"{item['customer_email']} {' '.join(item.get('keywords', []))}"
            for item in self.interactions
        ]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def _default_fallback_dataset(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "PI-101",
                "category": "Damaged / Defective Product",
                "customer_email": "My package arrived damaged.",
                "agent_reply": "We are sorry for the inconvenience. We will arrange a replacement immediately.",
                "keywords": ["damaged", "broken", "package"]
            },
            {
                "id": "PI-102",
                "category": "Delivery / Shipping Delay",
                "customer_email": "My order has not been delivered yet.",
                "agent_reply": "We apologize for the delay. Our team is checking your shipment. You will receive an update shortly.",
                "keywords": ["delivered", "delay", "shipping", "order"]
            }
        ]
