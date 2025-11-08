"""
Quick test script to run a single query through the pipeline.
"""

import sys
import os
import asyncio

# Ensure repo root is on sys.path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from rich.console import Console
from models.state import State
from scripts.test_necthrall_real import (
    initialize_system,
    build_wrapped_agents,
    AgentRecorder,
)
from orchestrator.graph import build_workflow
from types import SimpleNamespace
import time


async def test_query(query: str):
    console = Console()
    console.print(f"[bold cyan]Testing query:[/] {query}\n")

    # Initialize system
    console.print("Initializing system...")
    import main

    init_status = await initialize_system(console, debug=True)
    console.print(f"Init status: {init_status}\n")

    # Build workflow
    recorder = AgentRecorder()
    mock_request = SimpleNamespace(app=main.app)
    mock_agents = build_wrapped_agents(recorder, main.app)
    workflow = build_workflow(mock_request, mock_agents)

    # Run query
    console.print(f"[bold]Running query...[/]\n")
    initial_state = State(original_query=query)

    start = time.perf_counter()
    try:
        result = await workflow.ainvoke(initial_state)
        elapsed = time.perf_counter() - start

        # Handle both dict and State object returns
        if isinstance(result, dict):
            result_state = State(**result) if result else State()
        else:
            result_state = result

        console.print(f"\n[green]✓ Query completed in {elapsed:.2f}s[/]")
        console.print(f"\nPapers found: {len(result_state.papers_metadata or [])}")
        console.print(f"Papers selected: {len(result_state.filtered_papers or [])}")
        console.print(f"PDFs extracted: {len(result_state.pdf_contents or [])}")
        console.print(f"Citations: {len(result_state.citations or [])}")

        if result_state.synthesized_answer:
            console.print(f"\n[bold]Answer:[/]")
            console.print(result_state.synthesized_answer[:2500])

        console.print(f"\n[bold]Agent Steps:[/]")
        for step in recorder.steps:
            status = "✅" if step["success"] else "❌"
            console.print(f"  {status} {step['name']}: {step['time']:.2f}s")

        if recorder.errors:
            console.print(f"\n[bold red]Errors:[/]")
            for err in recorder.errors:
                console.print(
                    f"  {err['name']}: {err['error_type']} - {err['message']}"
                )

    except Exception as e:
        console.print(f"[red]✗ Query failed: {e}[/]")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Are persistent organic pollutants safe for the environment?"
    )
    asyncio.run(test_query(query))
