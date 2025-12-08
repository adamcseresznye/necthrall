"""Reusable UI components for Necthrall."""

from nicegui import ui
from loguru import logger


import re


def _clean_markdown(text: str) -> str:
    """Clean up markdown formatting for display.

    Removes list markers while preserving **bold** formatting.
    """
    if not text:
        return ""
    # Remove "* " but not "**" (bold) - match single * followed by space, not preceded by *
    text = re.sub(r"^(\s*)[-*](?!\s)", r"\1* ", text, flags=re.MULTILINE)
    return text


def render_loading():
    """Render the loading spinner with status text."""
    with ui.column().classes("loading-container w-full") as container:
        ui.spinner("dots", size="xl", color="#f97316")
        ui.label("Analyzing Research...").classes("loading-text")
        ui.label("Searching papers, extracting insights, synthesizing answer").classes(
            "text-slate-400 text-sm mt-2"
        )
    return container


def render_answer(result):
    """Render the answer section.

    Args:
        result: PipelineResult object with answer, execution_time, finalists, passages
    """
    with ui.column().classes("answer-section w-full"):
        # Stats row
        with ui.row().classes("items-center gap-3 mb-4"):
            with ui.row().classes("gap-2"):
                ui.label(f"⏱️ {result.execution_time:.1f}s").classes("stats-badge")
                ui.label(f"📄 {len(result.finalists)} papers analyzed").classes(
                    "stats-badge"
                )
                ui.label(f"📝 {len(result.passages)} sources cited").classes(
                    "stats-badge"
                )

        # Answer text - clean up markdown and display
        cleaned_answer = _clean_markdown(result.answer)
        ui.markdown(result.answer, extras=["tables"]).classes("answer-text")


def render_sources(passages):
    """Render the sources section as cards.

    Args:
        passages: List of passage objects with node and score attributes
    """
    if not passages:
        return

    with ui.column().classes("sources-section w-full"):
        ui.label("SOURCES CITED").classes("sources-title")

        for idx, passage in enumerate(passages):
            try:
                # Extract passage info
                text = (
                    passage.node.get_content()
                    if hasattr(passage.node, "get_content")
                    else str(passage.node)
                )
                metadata = (
                    passage.node.metadata if hasattr(passage.node, "metadata") else {}
                )
                paper_title = metadata.get("paper_title", "Unknown Source")
                paper_id = metadata.get("paper_id", "")
                section = metadata.get("section", "")
                # Try to get PDF URL from metadata
                pdf_url = metadata.get("pdf_url", "") or metadata.get("url", "")

                with ui.expansion().classes("source-card").props("dense") as expansion:
                    # Header slot for title
                    with expansion.add_slot("header"):
                        with ui.column().classes("w-full"):
                            # Title
                            ui.label(paper_title).classes("source-title")

                            # Meta info
                            meta_parts = []
                            if section:
                                meta_parts.append(section)
                            meta_parts.append(f"Citation [{idx + 1}]")
                            ui.label(" • ".join(meta_parts)).classes("source-meta")

                            # PDF link if available
                            if pdf_url:
                                with ui.row().classes("items-center mt-2"):
                                    with (
                                        ui.element("a")
                                        .props(f'href="{pdf_url}" target="_blank"')
                                        .classes("source-link-btn")
                                    ):
                                        ui.label("📄 Open Access PDF")
                            else:
                                with ui.row().classes("items-center mt-2"):
                                    ui.html(
                                        '<span class="source-link-badge">📄 PDF not available</span>',
                                        sanitize=False,
                                    )

                    # Expanded content - full passage
                    with ui.column().classes("passage-content"):
                        ui.label(text).classes("source-snippet")

            except Exception as e:
                logger.warning(f"Failed to render passage {idx}: {e}")
                continue


def render_error(error_message: str, error_stage: str = None):
    """Render an error card.

    Args:
        error_message: The error message to display
        error_stage: Optional stage where the error occurred
    """
    with ui.column().classes("error-card w-full"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("error_outline", color="#ef4444", size="sm")
            ui.label("Analysis Failed").classes("text-lg font-semibold text-red-600")
        ui.label(error_message).classes("text-red-500 mt-2")
        if error_stage:
            ui.label(f"Stage: {error_stage}").classes("text-red-400 text-sm mt-1")


def render_exception_error(exception: Exception):
    """Render an error card for an exception.

    Args:
        exception: The exception that was raised
    """
    with ui.column().classes("error-card w-full"):
        ui.icon("error_outline", color="#ef4444", size="sm")
        ui.label("Error").classes("text-lg font-semibold text-red-600")
        ui.label(str(exception)).classes("text-red-500 mt-2")
