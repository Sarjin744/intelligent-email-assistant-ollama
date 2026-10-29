"""
Comprehensive Automated Test Suite for Experiment 3
Course: 24CS2506 – Building Applications using GPT-4
"""

import sys
import os
import json
import time

# Ensure UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from engine import (
    OllamaClient,
    EmailAnalyzer,
    InteractionRetriever,
    EmailAnalytics,
    IntelligentEmailAssistant
)

def test_analyzer():
    print("[TEST 1/6] Testing EmailAnalyzer...")
    analyzer = EmailAnalyzer()

    # Case A: Shipping Delay
    res_a = analyzer.analyze("My order has not been delivered yet. Order #88219. Expected 3 days ago.")
    assert "Delivery" in res_a["intent"], f"Expected Delivery intent, got {res_a['intent']}"
    assert "88219" in res_a["context"]["order_ids"], "Failed to extract Order ID"
    assert res_a["intent_confidence"] > 0.6, "Low confidence"
    print("  ✓ Shipping Delay analysis verified.")

    # Case B: Broken Laptop Screen (Lab Sheet Example)
    res_b = analyzer.analyze("My laptop screen is broken and showing vertical lines after opening the box. Need replacement immediately!")
    assert "Damaged" in res_b["intent"], f"Expected Damaged intent, got {res_b['intent']}"
    assert "laptop" in res_b["context"]["products_mentioned"] or "screen" in res_b["context"]["products_mentioned"]
    assert "Urgent" in res_b["tone"] or res_b["context"]["urgency_level"] == "High / Escalated"
    print("  ✓ Broken Laptop Screen analysis verified.")

    # Case C: Angry / Frustrated Tone
    res_c = analyzer.analyze("THIS IS UNACCEPTABLE! Terrible service, worst experience ever! I want a refund now!")
    assert "Angry" in res_c["tone"], f"Expected Angry tone, got {res_c['tone']}"
    assert "Refund" in res_c["intent"], f"Expected Refund intent, got {res_c['intent']}"
    print("  ✓ Tone and Sentiment detection verified.")

def test_retriever():
    print("\n[TEST 2/6] Testing InteractionRetriever (Vector Memory & Cosine Similarity)...")
    retriever = InteractionRetriever()
    assert len(retriever.interactions) >= 8, "Dataset too small"
    assert retriever.tfidf_matrix is not None, "TF-IDF matrix not initialized"

    # Query 1: Laptop Screen Broken
    matches_1 = retriever.find_similar("My laptop screen is broken", top_k=2)
    assert len(matches_1) == 2, "Expected 2 matches"
    assert matches_1[0]["similarity_score"] > 0.4, f"Low similarity score: {matches_1[0]['similarity_score']}"
    assert "broken" in matches_1[0]["customer_email"].lower() or "screen" in matches_1[0]["customer_email"].lower()
    print(f"  ✓ Top Match 1: '{matches_1[0]['customer_email']}' (Score: {matches_1[0]['similarity_score']:.4f})")

    # Query 2: Package Arrived Damaged
    matches_2 = retriever.find_similar("My package arrived damaged.", top_k=1)
    assert matches_2[0]["similarity_score"] > 0.45, f"Expected > 0.45, got {matches_2[0]['similarity_score']}"
    assert "damaged" in matches_2[0]["customer_email"].lower()
    print(f"  ✓ Top Match 2: '{matches_2[0]['customer_email']}' (Score: {matches_2[0]['similarity_score']:.4f})")

    # Dynamic insertion test (using temporary copy to protect original dataset)
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    temp_json = os.path.join(temp_dir, "temp_interactions.json")
    shutil.copy("data/past_interactions.json", temp_json)
    
    temp_retriever = InteractionRetriever(data_path=temp_json)
    initial_count = len(temp_retriever.interactions)
    temp_retriever.add_interaction(
        customer_email="Can I exchange my t-shirt for a larger size?",
        agent_reply="Yes, we support free exchanges within 14 days.",
        category="Exchange",
        keywords=["exchange", "size", "shirt"]
    )
    assert len(temp_retriever.interactions) == initial_count + 1
    new_match = temp_retriever.find_similar("exchange my shirt size", top_k=1)
    assert "exchange" in new_match[0]["customer_email"].lower()
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("  ✓ Dynamic interaction learning & index retraining verified.")

def test_analytics():
    print("\n[TEST 3/6] Testing EmailAnalytics (Readability & Linguistic Scores)...")
    analytics = EmailAnalytics()

    sample_email = (
        "Dear Customer,\n\n"
        "We apologize for the inconvenience. We understand how frustrating this delay is. "
        "Our team has initiated an immediate replacement and you will receive tracking shortly.\n\n"
        "Warm regards,\nCustomer Support Team"
    )

    metrics = analytics.analyze(sample_email)
    assert metrics["word_count"] > 20, "Word count too low"
    assert metrics["sentence_count"] >= 3, "Sentence count mismatch"
    assert 0 <= metrics["reading_ease"] <= 100, f"Reading ease out of range: {metrics['reading_ease']}"
    assert metrics["politeness_score"] >= 80, f"Politeness score too low: {metrics['politeness_score']}"
    assert metrics["actionability_score"] >= 80, f"Actionability score too low: {metrics['actionability_score']}"
    assert metrics["professionalism_score"] >= 90, f"Professionalism score too low: {metrics['professionalism_score']}"
    print(f"  ✓ Analytics Computed: Ease={metrics['reading_ease']}, Politeness={metrics['politeness_score']}%, Actionability={metrics['actionability_score']}%, Prof={metrics['professionalism_score']}%")

def test_assistant_pipeline():
    print("\n[TEST 4/6] Testing IntelligentEmailAssistant Pipeline...")
    assistant = IntelligentEmailAssistant()

    # Few-Shot Mode
    res_few = assistant.generate_reply(
        customer_email="My laptop screen is broken.",
        customer_name="Alex Murphy",
        use_few_shot=True
    )
    assert res_few["generated_reply"], "Generated reply is empty"
    assert "Alex" in res_few["generated_reply"] or "Murphy" in res_few["generated_reply"] or "Customer" in res_few["generated_reply"]
    assert len(res_few["similar_interactions"]) > 0, "No similar interactions found in few-shot"
    print("  ✓ Few-Shot generation completed successfully.")

    # Zero-Shot Mode
    res_zero = assistant.generate_reply(
        customer_email="My laptop screen is broken.",
        customer_name="Alex Murphy",
        use_few_shot=False
    )
    assert res_zero["generated_reply"], "Zero-shot generated reply is empty"
    assert len(res_zero["similar_interactions"]) == 0, "Few-shot examples should be omitted"
    print("  ✓ Zero-Shot generation completed successfully.")

def test_cli_execution():
    print("\n[TEST 5/6] Testing Interactive CLI Execution Simulation...")
    import subprocess
    cmd = [sys.executable, "cli.py"]
    # Provide choice 2 (Broken laptop), then choice 8 (Exit)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    stdout, stderr = proc.communicate(input="2\n8\n", timeout=20)
    assert proc.returncode == 0, f"CLI exited with error code {proc.returncode}: {stderr}"
    assert "Damaged" in stdout and "Product" in stdout, "CLI failed to display identified intent"
    assert "Retrieved Previous Interactions" in stdout, "CLI failed to display retrieved interactions"
    assert "AI Suggested Reply" in stdout, "CLI failed to display AI suggested reply"
    print("  ✓ CLI executed with exit code 0 and all tables rendered.")

def test_batch_launchers():
    print("\n[TEST 6/6] Testing Batch Launcher Files...")
    assert os.path.exists("run_app.bat"), "run_app.bat missing"
    assert os.path.exists("run_cli.bat"), "run_cli.bat missing"
    with open("run_app.bat", "r") as f:
        content_app = f.read()
        assert "streamlit run app.py" in content_app, "run_app.bat missing streamlit command"
    with open("run_cli.bat", "r") as f:
        content_cli = f.read()
        assert "cli.py" in content_cli, "run_cli.bat missing cli.py command"
    print("  ✓ Batch launcher scripts verified.")

if __name__ == "__main__":
    start_total = time.time()
    print("=" * 65)
    print("🚀 RUNNING FULL SYSTEM SCAN & VALIDATION FOR EXPERIMENT 3")
    print("=" * 65)

    test_analyzer()
    test_retriever()
    test_analytics()
    test_assistant_pipeline()
    test_cli_execution()
    test_batch_launchers()

    elapsed = round(time.time() - start_total, 2)
    print("\n" + "=" * 65)
    print(f"🎉 ALL 6 TEST SUITES PASSED FLAWLESSLY IN {elapsed}s!")
    print("=" * 65)
