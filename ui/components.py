"""Reusable UI components for Necthrall."""

import json

from loguru import logger
from nicegui import ui


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

    def copy_content():
        """Construct and copy content to clipboard."""
        content = result.answer + "\n\n### Sources Cited\n"
        for idx, passage in enumerate(result.passages):
            metadata = getattr(passage, "metadata", {})
            title = metadata.get("title", "Unknown Source")
            url = metadata.get("url", "")
            content += f"{idx + 1}. {title} - {url}\n"

        # Serialize to JSON to handle quotes/newlines safely
        safe_content = json.dumps(content)
        ui.run_javascript(f"navigator.clipboard.writeText({safe_content})")
        ui.notify("Copied to clipboard", type="positive")

    with ui.column().classes("answer-section w-full"):
        # Stats row - responsive layout
        with ui.row().classes("w-full justify-between items-start mb-4 no-wrap"):
            # Stats badges container - allows wrapping internally
            with ui.row().classes("flex-wrap gap-2 items-center"):
                ui.label(f"⏱️ {result.execution_time:.1f}s").classes("stats-badge")
                ui.label(f"📄 {len(result.finalists)} papers analyzed").classes(
                    "stats-badge"
                )
                ui.label(f"📝 {len(result.passages)} sources cited").classes(
                    "stats-badge"
                )

            ui.button(icon="content_copy", on_click=copy_content).props(
                "flat round size=sm"
            ).classes("text-gray-500 shrink-0").tooltip("Copy to clipboard")

        # Answer text - clean up markdown and display
        ui.markdown(result.answer, extras=["tables"]).classes("w-full").style(
            "word-wrap: break-word; overflow-wrap: break-word; hyphens: auto; word-break: break-word; line-height: 1.8;"
        )


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
                text = getattr(passage, "text", "")
                metadata = getattr(passage, "metadata", {})
                paper_title = metadata.get("title", "Unknown Source")
                paper_id = getattr(passage, "paper_id", "")
                section = metadata.get("section", "")
                # Try to get PDF URL from metadata
                pdf_url = metadata.get("url", "")

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


class SearchProgress:
    """Vertical stepper to visualize search progress."""

    def __init__(self):
        with ui.column().classes("w-full items-center"):
            with ui.card().classes(
                "w-full max-w-md p-6 shadow-lg border border-slate-100"
            ):
                ui.label("Researching...").classes(
                    "text-xl font-bold text-slate-700 mb-4 text-center w-full"
                )
                self.stepper = (
                    ui.stepper().props("vertical animated flat").classes("w-full")
                )
                with self.stepper:
                    with ui.step("Refining Query"):
                        ui.label("Clarifying your research question...").classes(
                            "text-slate-600"
                        )

                    with ui.step("Searching Sources"):
                        ui.label("Scanning millions of open-access papers...").classes(
                            "text-slate-600"
                        )

                    with ui.step("Reading Documents"):
                        with ui.column().classes("gap-2"):
                            ui.label(
                                "Downloading and reading full-text PDFs..."
                            ).classes("text-slate-600")
                            ui.spinner("dots", size="lg", color="primary")

                    with ui.step("Analyzing Data"):
                        ui.label("Extracting key evidence and citations...").classes(
                            "text-slate-600"
                        )

                    with ui.step("Writing Answer"):
                        ui.label("Drafting your cited summary...").classes(
                            "text-slate-600"
                        )

    def next_step(self):
        """Advance to the next step."""
        self.stepper.next()
