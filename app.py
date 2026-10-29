"""
AI-Powered Intelligent Email Assistant using Ollama
Course: 24CS2506 – Building Applications using GPT-4
Experiment 3: Intelligent Email Assistant using Ollama
Karunya Institute of Technology and Sciences
"""

import json
import streamlit as st
import pandas as pd
import altair as alt

from engine import (
    IntelligentEmailAssistant,
    EmailAnalyzer,
    InteractionRetriever,
    EmailAnalytics
)

# Page configuration
st.set_page_config(
    page_title="Intelligent Email Assistant | Lab Exp 3",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0284c7;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 0.98rem;
        color: #64748b;
        margin-bottom: 18px;
    }
    .intent-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 8px;
    }
    .badge-intent { background-color: #e0f2fe; color: #0369a1; }
    .badge-tone { background-color: #fef3c7; color: #b45309; }
    .badge-urgency { background-color: #fee2e2; color: #991b1b; }
    .badge-entity { background-color: #f3e8ff; color: #6b21a8; }
    .card-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .reply-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 16px;
        font-size: 0.95rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "assistant" not in st.session_state:
    st.session_state.assistant = IntelligentEmailAssistant(model="llama3.2")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "history" not in st.session_state:
    st.session_state.history = []

assistant: IntelligentEmailAssistant = st.session_state.assistant

# Benchmark sample templates
PRESET_EMAILS = {
    "Select a preset email...": {
        "name": "",
        "subject": "",
        "body": ""
    },
    "📦 Delayed Delivery (Lab Sheet Ex 1)": {
        "name": "Sarah Connor",
        "subject": "Order delivery delayed - Order #88219",
        "body": "My order has not been delivered yet. It has been over 5 days since dispatch and the tracking has not updated."
    },
    "💻 Broken Laptop Screen (Lab Sheet Ex 3)": {
        "name": "Alex Murphy",
        "subject": "Laptop screen broken upon unboxing",
        "body": "My laptop screen is broken and showing horizontal lines. I just opened the package 10 minutes ago. Please replace it!"
    },
    "📦 Damaged Package (Lab Sheet Ex 2)": {
        "name": "Michael Scott",
        "subject": "Package arrived badly damaged",
        "body": "My package arrived damaged. The outer carton was torn and the product inside is completely crushed."
    },
    "💳 Refund Not Credited": {
        "name": "Jessica Pearson",
        "subject": "Refund pending for cancelled order #44910",
        "body": "I cancelled my order three days ago, but the refund amount has not credited to my bank account yet. Kindly assist."
    },
    "⚡ Duplicate Payment Debit": {
        "name": "Harvey Specter",
        "subject": "Charged twice on checkout",
        "body": "I was charged twice for the same transaction during checkout for order #9021. Kindly reverse the duplicate charge immediately."
    }
}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/email-open.png", width=95)
    st.markdown("### 🏛️ Karunya University")
    st.markdown("**24CS2506 – Building Applications using GPT-4**")
    st.markdown("**Experiment 3: Intelligent Email Assistant**")
    st.markdown("---")

    # Connection Status
    st.subheader("⚙️ Ollama Status")
    is_connected = assistant.ollama.check_connection()
    if is_connected:
        st.success("● Ollama Service Active (localhost:11434)")
        models = assistant.ollama.list_models()
        if models:
            selected_model = st.selectbox("Available Models", models, index=0)
            assistant.set_model(selected_model)
        else:
            st.info("No models currently loaded in Ollama.\nPull Llama 3.2: `ollama run llama3.2`")
    else:
        st.warning("○ Ollama Daemon Offline\n(Running in Semantic Fallback Engine)")

    st.markdown("---")
    st.subheader("🎛️ Assistant Settings")
    desired_tone = st.selectbox(
        "Response Persona / Tone",
        [
            "Empathetic & Professional",
            "Apologetic & Reassuring",
            "Formal & Concise",
            "Technical & Direct",
            "Urgent & Action-Oriented"
        ],
        index=0
    )

    use_few_shot = st.checkbox("Enable Previous Interactions (Few-Shot)", value=True, help="Condition response using historical high-quality email replies.")
    top_k_examples = st.slider("Historical Examples to Retrieve", min_value=1, max_value=3, value=2)

    st.markdown("---")
    st.caption("Lab Instructor: Dept of CSE | Karunya Institute of Technology & Sciences")

# Main Header
st.markdown('<div class="main-header">📧 Intelligent Email Assistant using Ollama</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Intent Understanding, Tone Detection, Context Extraction & Historical Few-Shot Reply Generation</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "✉️ Email Assistant Studio",
    "⚖️ Comparative Study (Zero vs Few-Shot)",
    "🧠 Vector Memory & Past Interactions",
    "📖 Lab Manual & Theory"
])

# ----------------------------------------------------
# TAB 1: STUDIO
# ----------------------------------------------------
with tab1:
    col_input, col_output = st.columns([1, 1], gap="medium")

    with col_input:
        st.markdown("#### 📥 Incoming Customer Email")

        preset_choice = st.selectbox("Load Benchmark Sample (From Lab Sheet)", list(PRESET_EMAILS.keys()))
        preset_data = PRESET_EMAILS[preset_choice]

        c_name = st.text_input("Customer Name", value=preset_data["name"] or "Valued Customer")
        c_subject = st.text_input("Subject Line", value=preset_data["subject"] or "Order Inquiry")
        c_body = st.text_area("Customer Email Body", value=preset_data["body"] or "My laptop screen is broken.", height=130)

        generate_btn = st.button("🚀 Analyze & Generate AI Reply", type="primary", use_container_width=True)

        if generate_btn:
            if not c_body.strip():
                st.error("Please enter a customer email message.")
            else:
                with st.spinner("Analyzing email & retrieving past interactions..."):
                    result = assistant.generate_reply(
                        customer_email=c_body,
                        desired_tone=desired_tone,
                        customer_name=c_name,
                        use_few_shot=use_few_shot,
                        top_k_examples=top_k_examples
                    )
                    st.session_state.last_result = result
                    st.session_state.history.append({
                        "customer": c_name,
                        "intent": result["analysis"]["intent"],
                        "tone": result["analysis"]["tone"],
                        "reply": result["generated_reply"]
                    })

    with col_output:
        st.markdown("#### 📤 AI Assistant Output & Analytics")

        if st.session_state.last_result:
            res = st.session_state.last_result
            analysis = res["analysis"]
            metrics = res["metrics"]
            ctx = analysis["context"]

            # Badges Display
            st.markdown(f"""
            <div>
                <span class="intent-badge badge-intent">🎯 Intent: {analysis['intent']} ({int(analysis['intent_confidence']*100)}%)</span>
                <span class="intent-badge badge-tone">🎭 Tone: {analysis['tone']}</span>
                <span class="intent-badge badge-urgency">⚡ Urgency: {ctx['urgency_level']}</span>
            </div>
            """, unsafe_allow_html=True)

            if ctx.get("order_ids") or ctx.get("products_mentioned"):
                entities_str = ""
                if ctx.get("order_ids"):
                    entities_str += f"Orders: {', '.join(ctx['order_ids'])} "
                if ctx.get("products_mentioned"):
                    entities_str += f"| Products: {', '.join(ctx['products_mentioned'])}"
                st.caption(f"📌 **Extracted Entities:** {entities_str}")

            # AI Reply Box
            st.markdown("##### ✉️ Suggested Professional Reply:")
            edited_reply = st.text_area(
                "Generated Reply (Editable)",
                value=res["generated_reply"],
                height=220,
                label_visibility="collapsed"
            )

            # Action bar
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col1:
                st.download_button(
                    "💾 Download .TXT",
                    data=edited_reply,
                    file_name=f"email_reply_{c_name.replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with btn_col2:
                export_json = json.dumps({
                    "customer": c_name,
                    "incoming_email": c_body,
                    "analysis": analysis,
                    "reply": edited_reply,
                    "metrics": metrics
                }, indent=2)
                st.download_button(
                    "📄 Export JSON",
                    data=export_json,
                    file_name=f"email_ticket_{c_name.replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with btn_col3:
                st.caption(f"⚡ Source: `{res['generation_source']}` ({res['execution_time_seconds']}s)")

            # Analytics Metric Cards
            st.markdown("---")
            st.markdown("##### 📊 Reply Quality & Readability Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Politeness Index", f"{metrics['politeness_score']}%", "Empathy & Etiquette")
            m2.metric("Actionability", f"{metrics['actionability_score']}%", "Clear Next Steps")
            m3.metric("Flesch Ease", f"{metrics['reading_ease']}", metrics["reading_label"])
            m4.metric("Word Count", f"{metrics['word_count']}", f"~{metrics['reading_time_seconds']}s read")

            # Expandable Past Interactions
            if res["similar_interactions"]:
                with st.expander("🧠 Retrieved Past Interactions Used in Conditioning (Few-Shot)"):
                    for idx, pi in enumerate(res["similar_interactions"], 1):
                        st.markdown(f"**Match #{idx} (Similarity: {pi['similarity_percentage']}% | Category: {pi['category']})**")
                        st.markdown(f"> *Past Email:* \"{pi['customer_email']}\"")
                        st.markdown(f"> *Past Approved Reply:* \"{pi['agent_reply']}\"")
                        st.markdown("---")

            # Expandable Prompt Inspector
            with st.expander("🔍 View Raw Prompt Sent to Model"):
                st.code(res["prompt_used"], language="text")

        else:
            st.info("👈 Select a sample email or type your own query, then click **'Analyze & Generate AI Reply'** to view the results.")

# ----------------------------------------------------
# TAB 2: COMPARATIVE STUDY (ZERO-SHOT VS FEW-SHOT)
# ----------------------------------------------------
with tab2:
    st.markdown("### ⚖️ Section 3 Evaluation: Why use Previous Interactions?")
    st.markdown("""
    According to the lab manual:
    > **Instead of replying randomly, AI looks at similar previous emails.**
    > - *Past Email:* "My package arrived damaged."
    > - *Previous Reply:* "We are sorry for the inconvenience. We will arrange a replacement immediately."
    > - *Current Email:* "My laptop screen is broken."
    > - *Suggested Reply:* "We apologize for the inconvenience. We will arrange a replacement after verification."
    """)

    st.markdown("---")
    compare_input = st.text_area(
        "Test Email for Comparative Evaluation",
        value="My laptop screen is broken and showing vertical lines after opening the box.",
        height=70
    )

    if st.button("⚡ Run Comparative Analysis", type="primary"):
        with st.spinner("Generating side-by-side comparison..."):
            zero_res = assistant.generate_reply(
                customer_email=compare_input,
                customer_name="Alex Murphy",
                use_few_shot=False
            )
            few_res = assistant.generate_reply(
                customer_email=compare_input,
                customer_name="Alex Murphy",
                use_few_shot=True
            )

            c1, c2 = st.columns(2)

            with c1:
                st.error("### ❌ Zero-Shot (Without Previous Interactions)")
                st.caption("Prompt contains only instructions without historical examples.")
                st.markdown(f'<div class="card-box">{zero_res["generated_reply"]}</div>', unsafe_allow_html=True)
                zm = zero_res["metrics"]
                st.metric("Actionability Score", f"{zm['actionability_score']}%")
                st.metric("Politeness Score", f"{zm['politeness_score']}%")
                st.metric("Professionalism", f"{zm['professionalism_score']}%")

            with c2:
                st.success("### ✅ Few-Shot (Grounded on Previous Interactions)")
                st.caption("Conditioned on top-ranked similar interactions via TF-IDF + Cosine Similarity.")
                st.markdown(f'<div class="card-box" style="border-color: #86efac; background-color: #f0fdf4;">{few_res["generated_reply"]}</div>', unsafe_allow_html=True)
                fm = few_res["metrics"]
                st.metric("Actionability Score", f"{fm['actionability_score']}%", f"+{fm['actionability_score'] - zm['actionability_score']}% vs Zero-Shot")
                st.metric("Politeness Score", f"{fm['politeness_score']}%", f"+{fm['politeness_score'] - zm['politeness_score']}% vs Zero-Shot")
                st.metric("Professionalism", f"{fm['professionalism_score']}%", f"+{fm['professionalism_score'] - zm['professionalism_score']}% vs Zero-Shot")

# ----------------------------------------------------
# TAB 3: VECTOR MEMORY & KNOWLEDGE BASE
# ----------------------------------------------------
with tab3:
    st.markdown("### 🧠 Vector Memory & Historical Interaction Repository")
    st.markdown("The system uses **TF-IDF Vectorization & Cosine Similarity** to match incoming customer emails to historical interactions.")

    retriever: InteractionRetriever = assistant.retriever

    st.markdown("#### 📚 Current Approved Interactions in Knowledge Base")
    df_interactions = pd.DataFrame(retriever.interactions)
    st.dataframe(df_interactions[["id", "category", "customer_email", "agent_reply"]], use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🧪 Test Similarity Matcher")
    test_q = st.text_input("Enter a query to test cosine similarity search:", "My parcel has not arrived yet")
    if test_q:
        sim_matches = retriever.find_similar(test_q, top_k=3)
        st.markdown(f"**Top Matches for:** `'{test_q}'`")
        for m in sim_matches:
            col_s1, col_s2 = st.columns([1, 4])
            with col_s1:
                st.progress(float(m["similarity_score"]))
                st.caption(f"Score: **{m['similarity_score']:.4f}** ({m['similarity_percentage']}%)")
            with col_s2:
                st.markdown(f"**{m['category']}** | `{m['customer_email']}`")
                st.markdown(f"*{m['agent_reply'][:110]}...*")

    st.markdown("---")
    st.markdown("#### ➕ Add New Historical Interaction")
    with st.expander("Expand to Add New Email Ticket to Vector Knowledge Base"):
        new_cat = st.text_input("Category", "Product Exchange")
        new_email = st.text_area("Past Customer Email")
        new_reply = st.text_area("Approved Agent Reply")
        new_kws = st.text_input("Keywords (comma separated)", "exchange, swap, size")

        if st.button("Save to Knowledge Base"):
            if new_email and new_reply:
                kw_list = [k.strip() for k in new_kws.split(",") if k.strip()]
                retriever.add_interaction(new_email, new_reply, category=new_cat, keywords=kw_list)
                st.success("New interaction added and TF-IDF memory index updated successfully!")
                st.rerun()
            else:
                st.warning("Please provide both customer email and agent reply.")

# ----------------------------------------------------
# TAB 4: LAB MANUAL & THEORY
# ----------------------------------------------------
with tab4:
    st.markdown("""
    ## 24CS2506 – Building Applications using GPT-4
    ### Experiment 3: Intelligent Email Assistant using Ollama

    ---

    ### 1. Aim
    To design and implement an **Intelligent Email Assistant** using local Large Language Models (**Meta Llama 3.2 via Ollama**), incorporating **Email Intent Recognition**, **Tone Detection**, **Context Extraction**, and **In-Context Learning (Few-Shot Retrieval)** using Previous Interactions.

    ---

    ### 2. Theoretical Concepts

    #### 2.1 What is an Email Assistant?
    An Email Assistant is an AI application that helps users read emails, understand customer queries, and automatically generate professional replies. Instead of writing responses manually, the AI analyzes the email and suggests an appropriate reply.

    *Example:*
    - **Customer Email:** *"My order has not been delivered yet."*
    - **AI Reply:** *"We apologize for the delay. Our team is checking your shipment. You will receive an update shortly."*

    #### 2.2 What is AI Reply Generation?
    AI Reply Generation is the process of generating suitable responses using Large Language Models.
    The AI understands:
    - **Customer intention:** Why is the customer contacting us? (e.g., Shipping Delay, Damaged Goods, Refund)
    - **Email tone:** Emotional mood of the customer (e.g., Angry, Frustrated, Urgent, Polite)
    - **Context:** Specific details such as Order IDs, product names, dates, and severity
    - **Previous conversations:** Historical precedent to maintain business policy and tone consistency

    #### 2.3 Why use Previous Interactions?
    Instead of replying randomly or generating hallucinated responses, AI looks at similar previous emails.
    - Grounding response generation in historical precedent prevents contradictory company promises.
    - Improves customer satisfaction through proven, empathetic resolutions.
    - Speeds up resolution time by leveraging approved business logic.

    *Example:*
    - **Past Email:** *"My package arrived damaged."*
    - **Previous Reply:** *"We are sorry for the inconvenience. We will arrange a replacement immediately."*
    - **Current Email:** *"My laptop screen is broken."*
    - **Suggested Reply:** *"We apologize for the inconvenience. We will arrange a replacement after verification."*

    ---

    ### 3. Mathematical Formulations

    #### 3.1 Cosine Similarity for Interaction Retrieval
    $$\\text{Cosine Similarity}(\\vec{q}, \\vec{d}) = \\frac{\\vec{q} \\cdot \\vec{d}}{\\|\\vec{q}\\| \\|\\vec{d}\\|} = \\frac{\\sum_{i=1}^n q_i d_i}{\\sqrt{\\sum_{i=1}^n q_i^2} \\sqrt{\\sum_{i=1}^n d_i^2}}$$

    #### 3.2 Flesch Reading Ease Formula
    $$\\text{Score} = 206.835 - 1.015 \\left(\\frac{\\text{Total Words}}{\\text{Total Sentences}}\\right) - 84.6 \\left(\\frac{\\text{Total Syllables}}{\\text{Total Words}}\\right)$$

    ---

    ### 4. Viva Voce Questions & Answers

    1. **Q: Why is Ollama preferred for building an enterprise Email Assistant?**  
       *A:* Ollama runs LLMs locally on-premise without transmitting confidential customer emails, order numbers, or personal data to third-party cloud servers, guaranteeing zero API costs and full GDPR/data privacy compliance.

    2. **Q: How does Few-Shot Retrieval differ from Fine-Tuning?**  
       *A:* Few-Shot Retrieval selects relevant examples dynamically at inference time and injects them into the prompt (In-Context Learning) without modifying model weights. Fine-Tuning permanently updates the weights of the model via gradient descent.

    3. **Q: What role does Tone Detection play in Customer Support AI?**  
       *A:* Tone detection allows the AI to de-escalate angry or frustrated customers by offering greater empathy and priority handling, while maintaining formal conciseness for straightforward logistical inquiries.
    """)
