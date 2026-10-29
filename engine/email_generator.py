"""
Core Email Generator Engine
Course: 24CS2506 – Building Applications using GPT-4
Experiment 3: Intelligent Email Assistant using Ollama
"""

import time
from typing import Dict, Any, List, Optional
from .ollama_client import OllamaClient
from .intent_analyzer import EmailAnalyzer
from .memory_retriever import InteractionRetriever
from .analytics import EmailAnalytics


class IntelligentEmailAssistant:
    """
    Coordinates email understanding, historical interaction retrieval (Few-Shot),
    prompt construction, LLM generation via Ollama (Llama 3.2), and content analytics.
    """
    def __init__(self, model: str = "llama3.2"):
        self.ollama = OllamaClient(default_model=model)
        self.analyzer = EmailAnalyzer()
        self.retriever = InteractionRetriever()
        self.evaluator = EmailAnalytics()
        self.current_model = model

    def set_model(self, model_name: str):
        self.current_model = model_name
        self.ollama.default_model = model_name

    def build_engineered_prompt(
        self,
        customer_email: str,
        analysis: Dict[str, Any],
        similar_interactions: List[Dict[str, Any]],
        desired_tone: str = "Empathetic & Professional",
        customer_name: Optional[str] = None,
        use_few_shot: bool = True
    ) -> str:
        """
        Constructs a structured prompt incorporating:
        - Customer details and original message
        - Detected Intent, Tone, and Context
        - Previous historical interactions (Few-Shot exemplars)
        - Reply requirements and persona instructions
        """
        greeting_name = customer_name if customer_name else "Valued Customer"

        prompt_parts = []
        prompt_parts.append("You are an Intelligent Customer Support Email Assistant.")
        prompt_parts.append("Draft a professional, clear, and empathetic email reply to the customer's query.\n")

        prompt_parts.append(f"CUSTOMER DETAILS:")
        prompt_parts.append(f"- Name: {greeting_name}")
        prompt_parts.append(f"- Identified Intent: {analysis['intent']}")
        prompt_parts.append(f"- Detected Customer Tone: {analysis['tone']}")
        prompt_parts.append(f"- Desired Assistant Tone: {desired_tone}")

        ctx = analysis.get("context", {})
        if ctx.get("order_ids"):
            prompt_parts.append(f"- Order ID(s) Mentioned: {', '.join(ctx['order_ids'])}")
        if ctx.get("products_mentioned"):
            prompt_parts.append(f"- Product(s): {', '.join(ctx['products_mentioned'])}")
        if ctx.get("urgency_level"):
            prompt_parts.append(f"- Urgency Level: {ctx['urgency_level']}")

        prompt_parts.append(f"\nCUSTOMER EMAIL MESSAGE:\n\"\"\"\n{customer_email}\n\"\"\"\n")

        if use_few_shot and similar_interactions:
            prompt_parts.append("=== RELEVANT PREVIOUS INTERACTIONS (REFERENCE EXAMPLES) ===")
            prompt_parts.append("Learn from these approved historical responses to maintain company policy, resolution speed, and professional tone:\n")
            for i, item in enumerate(similar_interactions, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Past Customer Email: \"{item['customer_email']}\"")
                prompt_parts.append(f"Approved Agent Reply:\n{item['agent_reply']}\n")
            prompt_parts.append("=== END REFERENCE EXAMPLES ===\n")

        prompt_parts.append("RESPONSE INSTRUCTIONS:")
        prompt_parts.append("1. Greet the customer warmly and professionally.")
        prompt_parts.append("2. Empathize with their issue directly and take ownership.")
        prompt_parts.append("3. Provide an actionable resolution, next step, or timeline (e.g., replacement dispatch, courier follow-up, verification, or refund process).")
        prompt_parts.append("4. Close with a courteous sign-off from Customer Support Team.")
        prompt_parts.append("5. Do not include meta-text or bracketed placeholders. Return ONLY the final email text ready to send.")

        return "\n".join(prompt_parts)

    def generate_reply(
        self,
        customer_email: str,
        desired_tone: str = "Empathetic & Professional",
        customer_name: Optional[str] = None,
        use_few_shot: bool = True,
        top_k_examples: int = 2
    ) -> Dict[str, Any]:
        """
        End-to-end processing:
        1. Analyze intent, tone, context
        2. Retrieve similar past interactions
        3. Build conditioned prompt
        4. Query Ollama (or semantic fallback)
        5. Evaluate response analytics
        """
        start_time = time.time()

        # Step 1: Semantic Email Analysis
        analysis = self.analyzer.analyze(customer_email)

        # Step 2: Retrieve Similar Past Interactions
        similar_interactions = []
        if use_few_shot:
            similar_interactions = self.retriever.find_similar(customer_email, top_k=top_k_examples)

        # Step 3: Build Prompt
        prompt = self.build_engineered_prompt(
            customer_email=customer_email,
            analysis=analysis,
            similar_interactions=similar_interactions,
            desired_tone=desired_tone,
            customer_name=customer_name,
            use_few_shot=use_few_shot
        )

        system_prompt = (
            "You are an expert customer service specialist for an enterprise company. "
            "Your replies are concise, compassionate, prompt, and strictly professional."
        )

        # Step 4: LLM Generation
        generation_source = "Ollama (Llama 3.2)"
        reply_text = None

        if self.ollama.check_connection():
            reply_text = self.ollama.generate(
                prompt=prompt,
                system=system_prompt,
                model=self.current_model,
                temperature=0.4,
                max_tokens=400
            )

        if not reply_text:
            generation_source = "Semantic Knowledge Fallback Engine"
            reply_text = self._generate_semantic_fallback(
                customer_email=customer_email,
                analysis=analysis,
                similar_interactions=similar_interactions,
                desired_tone=desired_tone,
                customer_name=customer_name
            )

        elapsed_time = round(time.time() - start_time, 2)

        # Step 5: Content Evaluation
        metrics = self.evaluator.analyze(reply_text)

        return {
            "customer_email": customer_email,
            "analysis": analysis,
            "similar_interactions": similar_interactions,
            "prompt_used": prompt,
            "generated_reply": reply_text,
            "generation_source": generation_source,
            "execution_time_seconds": elapsed_time,
            "metrics": metrics,
            "used_few_shot": use_few_shot
        }

    def _generate_semantic_fallback(
        self,
        customer_email: str,
        analysis: Dict[str, Any],
        similar_interactions: List[Dict[str, Any]],
        desired_tone: str,
        customer_name: Optional[str]
    ) -> str:
        """
        High-fidelity fallback generator when local Ollama daemon is inactive.
        Synthesizes historical best practices and syllabus examples.
        """
        intent = analysis["intent"]
        name = customer_name if customer_name else "Valued Customer"
        ctx = analysis["context"]
        orders = ctx.get("order_ids", [])
        order_ref = f" regarding order #{orders[0]}" if orders else ""

        # Check if top historical match exists
        top_match = similar_interactions[0] if similar_interactions else None

        if "Damaged" in intent or "broken" in customer_email.lower() or "screen" in customer_email.lower():
            if "screen" in customer_email.lower() or "laptop" in customer_email.lower():
                return (
                    f"Dear {name},\n\n"
                    f"We apologize for the inconvenience{order_ref}. We understand how concerning it is to receive a device with a broken display.\n\n"
                    f"We will arrange a replacement after quick verification. Kindly reply to this email with photos of the damaged screen and packaging so our team can approve an expedited swap within 24 hours.\n\n"
                    f"Thank you for your patience and cooperation.\n\n"
                    f"Warm regards,\n"
                    f"Customer Support & Hardware Care Team"
                )
            else:
                return (
                    f"Dear {name},\n\n"
                    f"We are truly sorry for the inconvenience caused{order_ref}. Receiving a damaged item is never the experience we want for you.\n\n"
                    f"We will arrange a complimentary replacement immediately. A return courier pickup will also be scheduled at your convenience to collect the damaged goods.\n\n"
                    f"Thank you for bringing this to our attention.\n\n"
                    f"Sincerely,\n"
                    f"Customer Fulfillment Support"
                )

        elif "Delivery" in intent or "delay" in customer_email.lower() or "delivered" in customer_email.lower():
            return (
                f"Dear {name},\n\n"
                f"We apologize for the delay in receiving your package{order_ref}. Our logistics team is actively checking your shipment with our courier partner.\n\n"
                f"You will receive an update shortly with an expedited delivery timeline and revised tracking details. We are monitoring this closely to ensure it reaches you without further holdup.\n\n"
                f"Warm regards,\n"
                f"Shipping & Logistics Operations"
            )

        elif "Refund" in intent or "cancel" in customer_email.lower():
            return (
                f"Dear {name},\n\n"
                f"Thank you for contacting us regarding your refund{order_ref}.\n\n"
                f"We have verified that the cancellation and refund transaction was successfully initiated. Depending on your bank's clearing cycle, the funds should reflect in your account within 3 to 5 business days.\n\n"
                f"Please let us know if you require a formal bank ARN number.\n\n"
                f"Best regards,\n"
                f"Billing & Accounts Desk"
            )

        elif "Billing" in intent or "charge" in customer_email.lower():
            return (
                f"Dear {name},\n\n"
                f"We apologize for the checkout billing anomaly{order_ref}.\n\n"
                f"Our finance team has reviewed the transactions and identified the duplicate charge. A full reversal has already been submitted to your payment provider, and the credited amount will appear on your statement within 24 to 48 hours.\n\n"
                f"Sincerely,\n"
                f"Finance & Payment Services"
            )

        else:
            return (
                f"Dear {name},\n\n"
                f"Thank you for reaching out to our support department{order_ref}.\n\n"
                f"We have reviewed your inquiry and our specialized support team is currently looking into the matter. We aim to provide you with a comprehensive resolution shortly.\n\n"
                f"If you have additional details to provide, please reply directly to this email.\n\n"
                f"Kind regards,\n"
                f"Customer Experience Team"
            )
