"""Page definitions for the Necthrall UI."""

from pathlib import Path
from nicegui import ui, context, app
from loguru import logger

from ui.styles import CUSTOM_CSS
from ui.components import (
    render_loading,
    render_answer,
    render_sources,
    render_error,
    render_exception_error,
)

# Path to logo directory
LOGO_DIR = (
    Path(__file__).resolve().parent.parent.joinpath("logo").joinpath("nechtrall.png")
)


def init_ui(fastapi_app):
    """Initialize the NiceGUI frontend.

    Args:
        fastapi_app: The FastAPI application instance with query_service in state
    """

    @ui.page("/")
    async def index_page():
        # Inject custom CSS
        ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")

        # Get client reference for connection checking
        client = context.client

        # State containers
        results_container = None

        def is_connected() -> bool:
            """Check if the client is still connected."""
            try:
                return client.has_socket_connection
            except Exception:
                return False

        async def handle_search():
            nonlocal results_container
            query_text = search_input.value.strip()

            if not query_text:
                ui.notify("Please enter a research question", type="warning")
                return

            # Clear previous results
            if results_container and is_connected():
                results_container.clear()

            # Show loading
            if not is_connected():
                return
            with results_container:
                render_loading()

            try:
                # Call the query service
                result = await fastapi_app.state.query_service.process_query(query_text)

                # Check if client is still connected before updating UI
                if not is_connected():
                    logger.info(
                        "Client disconnected during query processing, skipping UI update"
                    )
                    return

                # Clear loading
                results_container.clear()

                with results_container:
                    if result.success and result.answer:
                        # Success - show answer and sources
                        render_answer(result)
                        render_sources(result.passages)
                    else:
                        # Failed
                        error_msg = (
                            result.error_message or "An unexpected error occurred"
                        )
                        render_error(error_msg, result.error_stage)

            except Exception as e:
                logger.exception("Query processing failed")
                # Only update UI if client is still connected
                if not is_connected():
                    logger.info("Client disconnected, skipping error UI update")
                    return
                try:
                    results_container.clear()
                    with results_container:
                        render_exception_error(e)
                except RuntimeError:
                    # Client was deleted, silently ignore
                    pass

        # =====================================================================
        # HEADER
        # =====================================================================
        with ui.row().classes("header-container items-center justify-center"):
            # Logo and brand
            with ui.row().classes("items-center gap-1"):

                ui.image(LOGO_DIR).classes("w-20 h-20 object-contain").props(
                    "no-spinner"
                )
                ui.label("Nechtrall").classes("text-4xl font-bold text-slate-800")

        # =====================================================================
        # MAIN CONTENT
        # =====================================================================
        with ui.column().classes("w-full items-center px-4 py-8"):
            # Search container
            with ui.column().classes("search-container"):
                # Search wrapper
                with ui.row().classes("search-wrapper w-full items-center gap-2"):
                    ui.icon("search", size="sm").classes("text-slate-400 ml-3")
                    search_input = (
                        ui.input(
                            placeholder="cardiovascular effects of intermittent fasting"
                        )
                        .classes("search-input flex-grow")
                        .props("borderless dense")
                    )
                    search_input.on("keydown.enter", handle_search)

                    ui.button("Search", on_click=handle_search).classes(
                        "search-btn mr-1"
                    )

                # Example queries
                with ui.row().classes(
                    "example-queries gap-2 mt-4 flex-wrap justify-center"
                ):
                    ui.label("Try:").classes("text-slate-400 text-sm")

                    async def set_example(text: str):
                        search_input.value = text

                    for example in [
                        "cardiovascular effects of intermittent fasting",
                        "sleep deprivation and cognitive performance",
                        "gut microbiome and mental health",
                    ]:
                        ui.button(
                            example,
                            on_click=lambda e=example: set_example(e),
                        ).props("flat dense").classes("example-btn")

                # Results container
                results_container = ui.column().classes("w-full mt-6")
