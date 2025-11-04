#!/usr/bin/env python
"""
Interactive real-world testing console for Necthrall Lite.

Usage:
    python scripts/test_necthrall_real.py [--debug]

This script initializes the real system components (embedding model, LLM clients,
workflow) without launching the FastAPI server and allows interactive queries
to exercise the end-to-end RAG pipeline.

Notes:
- Uses the project `main.app` FastAPI app state to share models.
- Builds a LangGraph workflow via `orchestrator.graph.build_workflow` and injects
  lightweight wrappers to collect timing and intermediate outputs for debugging.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

try:
    import psutil
except Exception:
    psutil = None

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.traceback import install as install_rich_traceback

# Import real system components
from main import app, load_embedding_model, llm_client, LLMClient
from orchestrator.graph import build_workflow
from models.state import State

# Real agents
from agents.search import SearchAgent
from agents.acquisition import AcquisitionAgent
from agents.processing_agent import ProcessingAgent
from agents.analysis import AnalysisAgent
from agents.query_optimization_agent import QueryOptimizationAgent
from agents.deduplication_agent import DeduplicationAgent
from agents.filtering_agent import FilteringAgent
from agents.fallback_refinement_agent import FallbackRefinementAgent

from utils.logging_setup import setup_logging


def get_memory_mb() -> float:
    """Return current process memory in MB (best-effort)."""
    try:
        if psutil:
            p = psutil.Process(os.getpid())
            return p.memory_info().rss / 1024**2
        else:
            # Fallback: try memory_profiler if available
            from memory_profiler import memory_usage

            mem = memory_usage(-1, interval=0.1, timeout=1)
            return mem[-1] if mem else 0.0
    except Exception:
        return 0.0


class AgentRecorder:
    """Collects per-agent execution metadata for display and debugging."""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.api_counters: Dict[str, int] = {}

    def record(
        self,
        name: str,
        success: bool,
        elapsed: float,
        extra: Dict[str, Any] | None = None,
    ):
        self.steps.append(
            {"name": name, "success": success, "time": elapsed, "extra": extra or {}}
        )

    def record_error(self, name: str, exc: Exception):
        self.errors.append(
            {"name": name, "error_type": type(exc).__name__, "message": str(exc)}
        )
        self.api_counters[type(exc).__name__] = (
            self.api_counters.get(type(exc).__name__, 0) + 1
        )


def make_method_wrapper(fn: Callable, recorder: AgentRecorder, name: str) -> Callable:
    """Wrap a sync or async method to capture timing/errors.

    Returns a callable with same sync/async nature as the original.
    """

    if asyncio.iscoroutinefunction(fn):

        async def wrapped(state: State):
            start = time.perf_counter()
            try:
                res = await fn(state)
                elapsed = time.perf_counter() - start
                recorder.record(
                    name, True, elapsed, {"result_summary": _summarize_state(res)}
                )
                return res
            except Exception as e:
                elapsed = time.perf_counter() - start
                recorder.record(name, False, elapsed, {"error": str(e)})
                recorder.record_error(name, e)
                raise

        return wrapped
    else:

        def wrapped(state: State):
            start = time.perf_counter()
            try:
                res = fn(state)
                elapsed = time.perf_counter() - start
                recorder.record(
                    name, True, elapsed, {"result_summary": _summarize_state(res)}
                )
                return res
            except Exception as e:
                elapsed = time.perf_counter() - start
                recorder.record(name, False, elapsed, {"error": str(e)})
                recorder.record_error(name, e)
                raise

        return wrapped


def _summarize_state(state_or_val: Any) -> Dict[str, Any]:
    """Create a small summary dictionary for States or return values."""
    try:
        if isinstance(state_or_val, State):
            return {
                "papers": len(state_or_val.papers_metadata or []),
                "filtered": len(state_or_val.filtered_papers or []),
                "pdfs": len(state_or_val.pdf_contents or []),
            }
        elif hasattr(state_or_val, "__dict__"):
            return {"type": type(state_or_val).__name__}
        else:
            return {"type": type(state_or_val).__name__}
    except Exception:
        return {"type": "unserializable"}


def build_wrapped_agents(recorder: AgentRecorder, request_app) -> Dict[str, Any]:
    """Instantiate real agents and return a mock_agents dict of wrapped versions for build_workflow."""

    # Instantiate real agents
    qo = QueryOptimizationAgent()
    search = SearchAgent()
    dedup = DeduplicationAgent()
    filtering = FilteringAgent(SimpleNamespace(app=request_app))
    fallback = FallbackRefinementAgent()
    acquisition = AcquisitionAgent()
    processing = ProcessingAgent(request_app)
    analysis = AnalysisAgent()

    # Wrap methods used by workflow
    wrapped = {
        "query_optimizer": SimpleNamespace(
            optimize=make_method_wrapper(
                getattr(qo, "optimize"), recorder, "QueryOptimization"
            )
        ),
        "search_agent": SimpleNamespace(
            search=make_method_wrapper(
                getattr(search, "search"), recorder, "OpenAlexSearch"
            )
        ),
        "dedup_agent": SimpleNamespace(
            deduplicate=make_method_wrapper(
                getattr(dedup, "deduplicate"), recorder, "Deduplication"
            )
        ),
        "filtering_agent": SimpleNamespace(
            filter_candidates=make_method_wrapper(
                getattr(filtering, "filter_candidates"), recorder, "Filtering"
            )
        ),
        "fallback_refiner": SimpleNamespace(
            refine=make_method_wrapper(
                getattr(fallback, "refine"), recorder, "FallbackRefinement"
            )
        ),
        # acquisition is async callable
        "acquisition_agent": make_method_wrapper(
            acquisition.__call__, recorder, "Acquisition"
        ),
        # processing is sync callable instance - wrap __call__
        "processing_agent": make_method_wrapper(
            processing.__call__, recorder, "Processing"
        ),
        "analysis_agent": SimpleNamespace(
            analyze=make_method_wrapper(
                getattr(analysis, "analyze"), recorder, "Analysis"
            )
        ),
    }

    return wrapped


def pretty_print_results(
    console: Console, state: State, recorder: AgentRecorder, elapsed: float
):
    console.rule("SYNTHESIS RESULT")
    console.print(
        Panel(state.synthesized_answer or "<No answer>", title="Answer", expand=False)
    )

    # Citations
    table = Table(title="Citations", box=box.SIMPLE)
    table.add_column("#", style="cyan")
    table.add_column("paper_id", style="magenta")
    table.add_column("excerpt", overflow="fold")
    for i, cit in enumerate(state.citations or [], start=1):
        try:
            idx = cit.index if hasattr(cit, "index") else cit.get("index", i)
            pid = (
                cit.paper_id if hasattr(cit, "paper_id") else cit.get("paper_id", "N/A")
            )
            text = cit.text if hasattr(cit, "text") else cit.get("text", "")
        except Exception:
            pid = str(cit)
            text = ""
            idx = i
        table.add_row(str(idx), pid, text[:300])

    console.print(table)

    console.rule("PERFORMANCE")
    perf = Table(box=box.MINIMAL)
    perf.add_column("metric")
    perf.add_column("value")
    perf.add_row("total_time", f"{elapsed:.2f}s")
    perf.add_row("memory_mb", f"{get_memory_mb():.1f}")
    perf.add_row("papers_found", str(len(state.papers_metadata or [])))
    perf.add_row("papers_selected", str(len(state.filtered_papers or [])))
    perf.add_row("pdfs_extracted", str(len(state.pdf_contents or [])))
    perf.add_row("citations_returned", str(len(state.citations or [])))
    console.print(perf)

    console.rule("AGENT TIMINGS")
    at = Table(box=box.MINIMAL, show_header=True)
    at.add_column("step")
    at.add_column("success")
    at.add_column("time_s")
    at.add_column("notes")
    for s in recorder.steps:
        at.add_row(
            s["name"],
            "✅" if s["success"] else "❌",
            f"{s['time']:.2f}",
            json.dumps(s.get("extra", {})),
        )
    console.print(at)


async def initialize_system(console: Console, debug: bool = False) -> Dict[str, Any]:
    """Initialize embedding model and LLM clients into main.app state.

    Returns a dict containing status information.
    """
    status = {"embedding": False, "llm": False}

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()
    ) as progress:
        t1 = progress.add_task("Loading embedding models...", start=False)
        progress.start_task(t1)
        start = time.perf_counter()
        try:
            # Use existing startup routine
            await load_embedding_model()
            status["embedding"] = (
                hasattr(app.state, "embedding_model")
                and app.state.embedding_model is not None
            )
        except Exception as e:
            console.print(f"[red]Embedding model initialization failed: {e}")
        elapsed = time.perf_counter() - start
        progress.update(
            t1, description=f"Loading embedding models... ✅ ({elapsed:.1f}s)"
        )

        t2 = progress.add_task("Initializing LLM clients...", start=False)
        progress.start_task(t2)
        start = time.perf_counter()
        try:
            # Import and initialize LLM client (mimicking main.py startup logic)
            import main

            if main.llm_client is None:
                from utils.llm_client import LLMClient as _LLMClient

                main.llm_client = _LLMClient()
                main.LLMClient = _LLMClient
            status["llm"] = main.llm_client is not None
        except Exception as e:
            console.print(f"[red]LLM client initialization failed: {e}")
            if debug:
                import traceback

                traceback.print_exc()
        elapsed = time.perf_counter() - start
        progress.update(
            t2,
            description=f"Initializing LLM clients... {'✅' if status['llm'] else '⚠️'} ({elapsed:.1f}s)",
        )

    return status


def save_session_log(log_data: Dict[str, Any], prefix: str = "test_session") -> str:
    import datetime

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join("logs", f"{prefix}_{ts}.json")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)
    return fname


def main():
    install_rich_traceback()
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    # Setup logging
    setup_logging(development=args.debug)

    console = Console()
    console.clear()
    console.rule("🧬 NECTHRALL LITE - Real-World Testing Console")

    recorder = AgentRecorder()

    # Initialize system components
    init_start = time.perf_counter()
    console.print(
        "[bold]Initializing system components (embedding models, LLM clients)...[/]"
    )
    init_status = asyncio.run(initialize_system(console, debug=args.debug))
    init_elapsed = time.perf_counter() - init_start

    console.print(
        f"System init complete in {init_elapsed:.1f}s. Embeddings: {init_status['embedding']}, LLM: {init_status['llm']}"
    )

    # Build wrapped workflow
    mock_request = SimpleNamespace(app=app)
    mock_agents = build_wrapped_agents(recorder, app)
    console.print("Building workflow graph...")
    workflow = build_workflow(mock_request, mock_agents)
    console.print("Workflow ready. Enter queries (type 'help' for commands).")

    session_stats = {
        "queries": 0,
        "total_time": 0.0,
        "api_errors": {},
        "queries_detail": [],
    }

    try:
        while True:
            user = Prompt.ask("necthrall")
            cmd = user.strip()
            if not cmd:
                continue
            if cmd.lower() in ("quit", "exit"):
                console.print("👋 Goodbye! Saving session log...")
                fname = save_session_log(session_stats)
                console.print(f"Session saved to {fname}")
                break
            if cmd.lower() == "help":
                console.print("Commands: help, stats, debug, models, clear, quit")
                continue
            if cmd.lower() == "stats":
                q = session_stats["queries"]
                avg = session_stats["total_time"] / q if q else 0.0
                console.print(f"Queries processed: {q}, avg_time_s: {avg:.2f}")
                continue
            if cmd.lower() == "models":
                em = getattr(app.state, "embedding_model", None)
                console.print("Embedding model:", "loaded" if em else "not loaded")
                console.print("LLM client:", "ready" if llm_client else "not ready")
                continue
            if cmd.lower() == "clear":
                console.clear()
                continue
            if cmd.lower() == "debug":
                args.debug = not args.debug
                console.print(f"Debug mode set to {args.debug}")
                continue

            # Process scientific query
            console.print(f'\n🔍 Processing Query: "{cmd}"')
            initial_state = State(original_query=cmd)
            # Reset recorder steps for this query
            recorder.steps.clear()
            recorder.errors.clear()

            start = time.perf_counter()
            try:
                # The workflow contains async nodes (acquisition), so use ainvoke
                result_state = asyncio.run(workflow.ainvoke(initial_state))
            except Exception as e:
                console.print(f"[red]Workflow execution failed: {e}")
                recorder.record_error("workflow_invoke", e)
                result_state = None

            elapsed = time.perf_counter() - start
            session_stats["queries"] += 1
            session_stats["total_time"] += elapsed
            session_stats["queries_detail"].append(
                {"query": cmd, "time_s": elapsed, "errors": recorder.errors}
            )

            if result_state is None:
                console.print("No state returned due to error. See logs for details.")
                continue

            # Present results
            pretty_print_results(console, result_state, recorder, elapsed)

    except KeyboardInterrupt:
        console.print("Interrupted by user — saving session log and exiting.")
        save_session_log(session_stats)


if __name__ == "__main__":
    main()
