"""
Interactive CLI for Experiment 3 - Intelligent Email Assistant using Ollama
Course: 24CS2506 – Building Applications using GPT-4
Karunya Institute of Technology and Sciences
"""

import sys
import os

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm

from engine import IntelligentEmailAssistant

console = Console()

SAMPLE_EMAILS = [
    {
        "label": "Delivery Delay (From Lab Sheet)",
        "sender": "John Doe",
        "email": "My order has not been delivered yet. It has been over 5 days since dispatch. Order #8921."
    },
    {
        "label": "Broken Laptop Screen (From Lab Sheet)",
        "sender": "Rachel Green",
        "email": "My laptop screen is broken and showing vertical lines after opening the box. Need help ASAP!"
    },
    {
        "label": "Damaged Package (From Lab Sheet)",
        "sender": "Michael Scott",
        "email": "My package arrived damaged. The outer carton was torn and the product inside is completely crushed."
    },
    {
        "label": "Refund Not Received",
        "sender": "Sarah Jenkins",
        "email": "I cancelled order #44102 three days ago, but the refund amount has not credited to my bank account yet."
    },
    {
        "label": "Duplicate Payment Charge",
        "sender": "David Miller",
        "email": "I was charged twice for the same transaction during checkout. Kindly reverse the duplicate payment."
    }
]


def print_banner(assistant: IntelligentEmailAssistant):
    title = Text("📧  INTELLIGENT EMAIL ASSISTANT USING OLLAMA", style="bold cyan")
    subtitle = Text(
        "Course: 24CS2506 – Building Applications using GPT-4 | Experiment 3\n"
        "Department of Computer Science & Engineering | Karunya University",
        style="dim"
    )

    is_online = assistant.ollama.check_connection()
    ollama_status = (
        f"[bold green]● Ollama Connected[/bold green] (Model: {assistant.current_model})"
        if is_online
        else "[bold yellow]○ Ollama Offline[/bold yellow] (Using Local Semantic Knowledge Fallback)"
    )

    banner_content = Text.from_markup(
        f"{title}\n{subtitle}\n\n"
        f"[bold white]Engine Status:[/bold white] {ollama_status}\n"
        f"[dim]Core Pillars: Intent Recognition | Tone Analysis | Context Extraction | Few-Shot Previous Interactions[/dim]"
    )
    console.print(Panel(banner_content, border_style="cyan"))


def display_analysis_table(analysis: dict):
    table = Table(title="🔍 Email Semantic Analysis", border_style="bright_blue")
    table.add_column("Dimension", style="bold cyan")
    table.add_column("Extracted Value", style="bold white")
    table.add_column("Confidence / Detail", style="dim")

    table.add_row(
        "Customer Intention",
        analysis["intent"],
        f"Confidence: {int(analysis['intent_confidence']*100)}%"
    )
    table.add_row(
        "Email Tone / Sentiment",
        analysis["tone"],
        f"Confidence: {int(analysis['tone_confidence']*100)}%"
    )

    ctx = analysis.get("context", {})
    orders = ", ".join(ctx.get("order_ids", [])) or "None detected"
    products = ", ".join(ctx.get("products_mentioned", [])) or "None specified"
    urgency = ctx.get("urgency_level", "Standard")

    table.add_row("Order Reference(s)", orders, "Context Entity")
    table.add_row("Products Mentioned", products, "Context Entity")
    table.add_row("Urgency Level", urgency, "Triage Classification")

    console.print(table)


def display_retrieval_table(interactions: list):
    table = Table(title="🧠 Retrieved Previous Interactions (Cosine Similarity Matching)", border_style="green")
    table.add_column("#", style="bold green", width=3)
    table.add_column("Similarity", style="bold yellow", width=12)
    table.add_column("Past Customer Email", style="white", width=32)
    table.add_column("Approved Historical Reply", style="dim", width=42)

    for i, item in enumerate(interactions, 1):
        score_str = f"{item['similarity_percentage']}% ({item['similarity_score']:.3f})"
        table.add_row(
            str(i),
            score_str,
            item["customer_email"],
            item["agent_reply"][:100] + "..." if len(item["agent_reply"]) > 100 else item["agent_reply"]
        )

    console.print(table)


def display_metrics_table(metrics: dict, source: str, exec_time: float):
    table = Table(title="📊 Email Quality & Readability Metrics", border_style="magenta")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Score / Value", style="bold white")
    table.add_column("Benchmark / Standard", style="dim")

    table.add_row("Word Count", str(metrics["word_count"]), f"{metrics['words_per_sentence']} words/sentence")
    table.add_row("Reading Ease", f"{metrics['reading_ease']} / 100", metrics["reading_label"])
    table.add_row("Flesch-Kincaid Grade", f"Grade {metrics['grade_level']}", "Comprehension Accessibility")
    table.add_row("Politeness Score", f"{metrics['politeness_score']} / 100", "Empathy & Etiquette Index")
    table.add_row("Actionability Score", f"{metrics['actionability_score']} / 100", "Problem Resolution Index")
    table.add_row("Professionalism Score", f"{metrics['professionalism_score']} / 100", "Greeting, Structure & Sign-off")
    table.add_row("Generation Time", f"{exec_time}s", f"Source: {source}")

    console.print(table)


def run_comparison(assistant: IntelligentEmailAssistant, customer_email: str, customer_name: str):
    console.print("\n[bold yellow]⚡ Comparing Zero-Shot (No Past Interactions) vs Few-Shot (With Past Interactions)...[/bold yellow]")

    # Zero-shot
    res_zero = assistant.generate_reply(
        customer_email=customer_email,
        customer_name=customer_name,
        use_few_shot=False
    )
    # Few-shot
    res_few = assistant.generate_reply(
        customer_email=customer_email,
        customer_name=customer_name,
        use_few_shot=True
    )

    console.print(Panel(
        f"[bold red]❌ Zero-Shot Reply (Without Previous Interactions):[/bold red]\n\n{res_zero['generated_reply']}\n\n"
        f"[dim]Actionability: {res_zero['metrics']['actionability_score']}% | Professionalism: {res_zero['metrics']['professionalism_score']}%[/dim]",
        title="Zero-Shot Response",
        border_style="red"
    ))

    console.print(Panel(
        f"[bold green]✅ Few-Shot Reply (Grounded on Similar Past Interactions):[/bold green]\n\n{res_few['generated_reply']}\n\n"
        f"[dim]Actionability: {res_few['metrics']['actionability_score']}% | Professionalism: {res_few['metrics']['professionalism_score']}%[/dim]",
        title="Few-Shot Response (Knowledge-Guided)",
        border_style="green"
    ))


def main():
    assistant = IntelligentEmailAssistant()
    print_banner(assistant)

    while True:
        console.print("\n[bold cyan]Select an Option:[/bold cyan]")
        console.print("  [1] Benchmark 1: Delivery Delay ('My order has not been delivered yet')")
        console.print("  [2] Benchmark 2: Broken Laptop Screen ('My laptop screen is broken')")
        console.print("  [3] Benchmark 3: Damaged Package ('My package arrived damaged')")
        console.print("  [4] Benchmark 4: Refund & Cancellation Query")
        console.print("  [5] Benchmark 5: Duplicate Payment Query")
        console.print("  [6] Enter Custom Customer Email")
        console.print("  [7] Compare Zero-Shot vs Few-Shot (Lab Section 3 Experiment)")
        console.print("  [8] Exit")

        choice = Prompt.ask("\nEnter choice [1-8]", default="2")

        if choice == "8":
            console.print("[cyan]Exiting Intelligent Email Assistant. Happy learning![/cyan]")
            break

        if choice in ["1", "2", "3", "4", "5"]:
            idx = int(choice) - 1
            item = SAMPLE_EMAILS[idx]
            customer_email = item["email"]
            customer_name = item["sender"]
        elif choice == "6":
            customer_name = Prompt.ask("Customer Name", default="Customer")
            customer_email = Prompt.ask("Enter customer email text")
            if not customer_email.strip():
                console.print("[red]Email cannot be empty.[/red]")
                continue
        elif choice == "7":
            test_email = Prompt.ask(
                "Enter test email",
                default="My laptop screen is broken and won't turn on. Order #7829."
            )
            run_comparison(assistant, test_email, "Alex")
            continue
        else:
            console.print("[red]Invalid selection.[/red]")
            continue

        console.print(f"\n[bold white]Analyzing query for customer:[/bold white] [green]{customer_name}[/green]")
        console.print(Panel(customer_email, title="📥 Incoming Customer Email", border_style="blue"))

        # Generate response
        result = assistant.generate_reply(
            customer_email=customer_email,
            customer_name=customer_name,
            use_few_shot=True
        )

        # Display Semantic Analysis
        display_analysis_table(result["analysis"])

        # Display Retrieved Historical Interactions
        if result["similar_interactions"]:
            display_retrieval_table(result["similar_interactions"])

        # Display AI Generated Reply
        console.print(Panel(
            result["generated_reply"],
            title=f"📤 AI Suggested Reply ({result['generation_source']})",
            border_style="bold green"
        ))

        # Display Metrics
        display_metrics_table(
            result["metrics"],
            result["generation_source"],
            result["execution_time_seconds"]
        )


if __name__ == "__main__":
    main()
