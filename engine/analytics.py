"""
Content Analytics & Readability Metrics for Email Assistant
Course: 24CS2506 – Building Applications using GPT-4
Experiment 3: Intelligent Email Assistant using Ollama
"""

import re
from typing import Dict, Any, List


class EmailAnalytics:
    """
    Computes quantitative evaluation metrics for generated email responses:
    - Flesch Reading Ease & Grade Level
    - Professionalism & Formatting Index
    - Politeness & Empathy Index
    - Actionability / Solution Score
    """

    POLITE_KEYWORDS = [
        "apologize", "sorry", "inconvenience", "appreciate", "understand",
        "please", "kindly", "happy to help", "glad to assist", "thank you",
        "warm regards", "sincerely", "best regards"
    ]

    ACTION_KEYWORDS = [
        "replace", "replacement", "refund", "checking", "dispatch", "pickup",
        "verify", "verification", "investigate", "initiated", "attached",
        "link", "tracking", "shortly", "contact", "support"
    ]

    def analyze(self, email_text: str) -> Dict[str, Any]:
        """Compute full suite of email response metrics."""
        text = email_text.strip()
        words = re.findall(r"\b[A-Za-z0-9\-']+\b", text)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        total_words = len(words)
        total_sentences = max(len(sentences), 1)

        total_syllables = sum(self._count_syllables(w) for w in words)

        # Flesch Reading Ease
        words_per_sentence = total_words / total_sentences
        syllables_per_word = total_syllables / max(total_words, 1)

        flesch_reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        flesch_reading_ease = max(0.0, min(100.0, flesch_reading_ease))

        # Flesch-Kincaid Grade Level
        fk_grade = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
        fk_grade = max(1.0, min(18.0, fk_grade))

        # Qualitative Reading Interpretation
        if flesch_reading_ease >= 80:
            readability_label = "Very Easy / Plain English"
        elif flesch_reading_ease >= 60:
            readability_label = "Standard / Accessible"
        elif flesch_reading_ease >= 40:
            readability_label = "Moderately Complex"
        else:
            readability_label = "Difficult / Academic"

        # Politeness & Empathy Score
        lower_text = text.lower()
        polite_hits = sum(1 for kw in self.POLITE_KEYWORDS if kw in lower_text)
        politeness_score = min(100, int((polite_hits / 3) * 100))

        # Actionability Score
        action_hits = sum(1 for kw in self.ACTION_KEYWORDS if kw in lower_text)
        actionability_score = min(100, int((action_hits / 2) * 100))

        # Professionalism Score
        has_salutation = bool(re.search(r"\b(dear|hello|hi|greetings)\b", lower_text))
        has_signoff = bool(re.search(r"\b(regards|sincerely|warmly|best|support team)\b", lower_text))
        has_paragraphs = "\n" in text or len(sentences) >= 2

        prof_points = 40
        if has_salutation:
            prof_points += 25
        if has_signoff:
            prof_points += 20
        if has_paragraphs:
            prof_points += 15
        professionalism_score = min(100, prof_points)

        reading_time_sec = round((total_words / 200) * 60, 1)

        return {
            "word_count": total_words,
            "sentence_count": total_sentences,
            "words_per_sentence": round(words_per_sentence, 1),
            "reading_ease": round(flesch_reading_ease, 1),
            "reading_label": readability_label,
            "grade_level": round(fk_grade, 1),
            "politeness_score": politeness_score,
            "actionability_score": actionability_score,
            "professionalism_score": professionalism_score,
            "reading_time_seconds": reading_time_sec
        }

    def _count_syllables(self, word: str) -> int:
        w = word.lower()
        if len(w) <= 3:
            return 1
        w = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", w)
        w = re.sub(r"^y", "", w)
        matches = re.findall(r"[aeiouy]{1,2}", w)
        return max(1, len(matches))
