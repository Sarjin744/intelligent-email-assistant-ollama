"""
Intent, Tone, and Context Analyzer for Email Assistant
Course: 24CS2506 – Building Applications using GPT-4
Experiment 3: Intelligent Email Assistant using Ollama
"""

import re
from typing import Dict, Any, List


class EmailAnalyzer:
    """
    Analyzes customer emails to extract:
    1. Customer Intention (Intent classification)
    2. Email Tone (Emotional sentiment / urgency)
    3. Context (Order ID, products mentioned, temporal markers, severity)
    """

    INTENT_RULES = {
        "Delivery / Shipping Delay": [
            r"deliver", r"late", r"delay", r"track", r"where is my", r"shipment",
            r"courier", r"parcel", r"not arrived", r"haven't received", r"waiting for",
            r"eta", r"dispatch", r"expected date"
        ],
        "Damaged / Defective Product": [
            r"damaged", r"broken", r"crushed", r"torn", r"screen", r"shattered",
            r"faulty", r"defect", r"cracked", r"not working", r"malfunction",
            r"scratched", r"dented", r"hardware failure"
        ],
        "Refund & Cancellation": [
            r"refund", r"cancel", r"return", r"money back", r"bank", r"credit",
            r"reverse payment", r"reimbursement", r"compensat"
        ],
        "Billing & Payment Query": [
            r"charged twice", r"double charge", r"overcharge", r"invoice", r"receipt",
            r"billing", r"transaction fee", r"deducted", r"checkout error"
        ],
        "Wrong Item Received": [
            r"wrong item", r"wrong product", r"different item", r"received wired",
            r"incorrect item", r"not what i ordered", r"received instead"
        ],
        "Account & Authentication": [
            r"password", r"login", r"locked out", r"reset", r"verification code",
            r"authentication", r"otp", r"account access"
        ],
        "Warranty & Technical Inquiry": [
            r"warranty", r"guarantee", r"stopped turning on", r"repair", r"service center",
            r"support", r"replacement policy", r"technician"
        ]
    }

    TONE_RULES = {
        "Angry / Frustrated": [
            r"terrible", r"horrible", r"worst", r"unacceptable", r"furious", r"angry",
            r"ridiculous", r"pathetic", r"scam", r"lawyer", r"fraud", r"sue", r"complaint",
            r"!{2,}", r"\b(never again|disgusted)\b"
        ],
        "Urgent / Time-Sensitive": [
            r"immediately", r"urgent", r"asap", r"emergency", r"today itself",
            r"critical", r"without delay", r"right now", r"promptly"
        ],
        "Disappointed / Anxious": [
            r"sad", r"disappointed", r"worried", r"concerned", r"unhappy",
            r"let down", r"frustrated", r"inconvenience"
        ],
        "Polite / Neutral Inquiry": [
            r"please", r"kindly", r"thank", r"could you", r"would like to",
            r"inquiring", r"wondering", r"appreciate", r"regards"
        ]
    }

    def analyze(self, email_text: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of customer email."""
        text = email_text.strip()
        lower = text.lower()

        intent, intent_score = self._classify_intent(lower)
        tone, tone_score = self._detect_tone(text, lower)
        context = self._extract_context(text, lower)

        return {
            "intent": intent,
            "intent_confidence": round(intent_score, 2),
            "tone": tone,
            "tone_confidence": round(tone_score, 2),
            "context": context,
            "summary": f"Detected '{intent}' with '{tone}' tone."
        }

    def _classify_intent(self, text: str) -> (str, float):
        scores = {}
        for category, patterns in self.INTENT_RULES.items():
            match_count = sum(1 for p in patterns if re.search(p, text))
            scores[category] = match_count

        max_cat = max(scores, key=scores.get)
        max_val = scores[max_cat]
        if max_val == 0:
            return "General Customer Inquiry", 0.50

        # Confidence normalized
        confidence = min(0.60 + (max_val * 0.12), 0.98)
        return max_cat, confidence

    def _detect_tone(self, original_text: str, lower_text: str) -> (str, float):
        scores = {}
        for tone_name, patterns in self.TONE_RULES.items():
            match_count = sum(1 for p in patterns if re.search(p, lower_text))
            scores[tone_name] = match_count

        # Check for ALL CAPS shouting
        caps_words = [w for w in original_text.split() if w.isupper() and len(w) > 3]
        if len(caps_words) >= 2:
            scores["Angry / Frustrated"] = scores.get("Angry / Frustrated", 0) + 2

        # Check exclamation marks
        if "!" in original_text:
            scores["Urgent / Time-Sensitive"] = scores.get("Urgent / Time-Sensitive", 0) + 1

        best_tone = max(scores, key=scores.get)
        if scores[best_tone] == 0:
            return "Polite / Neutral Inquiry", 0.65

        confidence = min(0.60 + (scores[best_tone] * 0.15), 0.96)
        return best_tone, confidence

    def _extract_context(self, text: str, lower_text: str) -> Dict[str, Any]:
        # Extract Order Numbers (e.g., #12345, ORDER-987, ORD123)
        order_ids = re.findall(r"(?:order\s*#?|ord\s*#?|#)\s*([a-zA-Z0-9\-_]{4,15})", text, re.IGNORECASE)

        # Extract items / products
        common_products = [
            "laptop", "screen", "monitor", "phone", "headphones", "earphones",
            "charger", "watch", "smartwatch", "keyboard", "mouse", "package",
            "parcel", "carton", "tablet", "cable", "shoes", "jacket"
        ]
        products_found = [p for p in common_products if p in lower_text]

        # Extract time expressions
        time_expressions = re.findall(r"\b(\d+\s*(?:days?|hours?|weeks?|months?)\s*(?:ago|delay)?)\b", lower_text)

        # Detect urgency level
        urgency = "Standard"
        if any(w in lower_text for w in ["urgent", "asap", "immediately", "today", "emergency"]):
            urgency = "High / Escalated"
        elif any(w in lower_text for w in ["angry", "ridiculous", "unacceptable", "terrible"]):
            urgency = "High / Escalated"

        return {
            "order_ids": list(set(order_ids)),
            "products_mentioned": list(set(products_found)),
            "time_references": list(set(time_expressions)),
            "urgency_level": urgency
        }
