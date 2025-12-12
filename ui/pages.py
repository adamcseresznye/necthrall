"""Page definitions for the Necthrall UI."""

import asyncio
from pathlib import Path
from nicegui import ui, context, app
from loguru import logger
import datetime

from ui.styles import CUSTOM_CSS
from ui.components import (
    render_loading,
    render_answer,
    render_sources,
    render_error,
    render_exception_error,
    SearchProgress,
)
from ui.policies import PRIVACY_POLICY, TERMS_OF_SERVICE
from ui.constants import POSTHOG_SCRIPT, BUY_ME_COFFEE_WIDGET

# Path to logo directory
LOGO_DIR = (
    Path(__file__).resolve().parent.parent.joinpath("logo").joinpath("necthrall.png")
)


def init_ui(fastapi_app):
    """Initialize the NiceGUI frontend.

    Args:
        fastapi_app: The FastAPI application instance with query_service in state
    """

    @ui.page("/")
    async def index_page():
        # Inject PostHog analytics script
        ui.add_head_html(f"<script>{POSTHOG_SCRIPT}</script>")

        # Inject custom CSS
        ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")

        # Inject Buy Me A Coffee widget
        ui.add_body_html(BUY_ME_COFFEE_WIDGET)

        # Get client reference for connection checking
        client = context.client

        # =====================================================================
        # DIALOG: HOW IT WORKS
        # =====================================================================
        with ui.dialog() as about_dialog, ui.card().classes("q-pa-md w-full max-w-lg"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("How Necthrall Works").classes("text-xl font-bold")
                ui.button(icon="close", on_click=about_dialog.close).props(
                    "flat round dense"
                )

            with ui.column().classes(
                "gap-4 mt-4"
            ):  # Increased gap slightly for readability
                # Step 1: Highlight "Open Access" (Your data source)
                with ui.row().classes("items-start md:items-center gap-3"):
                    ui.icon("travel_explore").classes(
                        "text-blue-500 shrink-0 text-xl md:text-2xl"
                    )
                    ui.label(
                        "1. Search: We scan millions of open-access scientific papers."
                    ).classes("text-sm md:text-base break-words flex-1")

                # Step 2: Highlight "Credibility" (Your composite score: relevance + impact + authority)
                with ui.row().classes("items-start md:items-center gap-3"):
                    ui.icon("filter_alt").classes(
                        "text-green-500 shrink-0 text-xl md:text-2xl"
                    )
                    ui.label(
                        "2. Filter: We rank results by semantic relevance, credibility and citation impact."
                    ).classes("text-sm md:text-base break-words flex-1")

                # Step 3: Highlight "Full Text" & "Inline Citations" (Your key V3 features)
                with ui.row().classes("items-start md:items-center gap-3"):
                    ui.icon("auto_awesome").classes(
                        "text-purple-500 shrink-0 text-xl md:text-2xl"
                    )
                    ui.label(
                        "3. Synthesize: AI analyzes the full PDF text and writes a summary with inline citations."
                    ).classes("text-sm md:text-base break-words flex-1")

        # =====================================================================
        # DIALOG: PRIVACY POLICY
        # =====================================================================
        with (
            ui.dialog() as privacy_dialog,
            ui.card().classes("w-full max-w-lg q-pa-lg"),
        ):
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label("Privacy Policy").classes("text-xl font-bold")
                ui.button(icon="close", on_click=privacy_dialog.close).props(
                    "flat round dense"
                )

            with ui.scroll_area().classes("h-64 pr-4"):
                ui.markdown(PRIVACY_POLICY).classes("text-slate-700")

        # =====================================================================
        # DIALOG: TERMS OF SERVICE
        # =====================================================================
        with ui.dialog() as terms_dialog, ui.card().classes("w-full max-w-lg q-pa-lg"):
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label("Terms of Service").classes("text-xl font-bold")
                ui.button(icon="close", on_click=terms_dialog.close).props(
                    "flat round dense"
                )

            with ui.scroll_area().classes("h-64 pr-4"):
                ui.markdown(TERMS_OF_SERVICE).classes("text-slate-700")

        # State containers
        results_container = None
        example_queries_row = None
        search_container_ui = None

        def is_connected() -> bool:
            """Check if the client is still connected (soft check)."""
            try:
                # Only return False if explicitly disconnected, not transient issues
                return not getattr(client, "is_deleted", False)
            except Exception:
                # Assume connected on exception (avoid spurious errors)
                return True

        async def handle_search():
            nonlocal results_container
            query_text = search_input.value.strip()

            if not query_text:
                ui.notify("Please enter a research question", type="warning")
                return

            # Clear previous results
            if results_container:
                try:
                    results_container.clear()
                except RuntimeError:
                    # Client deleted, exit gracefully
                    return

            # Hide search container to focus on progress
            if search_container_ui:
                search_container_ui.set_visibility(False)

            # Hide the example queries immediately so they vanish during loading
            if example_queries_row:
                example_queries_row.set_visibility(False)

            # Show progress stepper
            progress_ui = None
            with results_container:
                progress_ui = SearchProgress()

            # Async callback to ensure UI updates
            async def advance_progress():
                progress_ui.next_step()
                # Force UI update by yielding control
                await asyncio.sleep(0)

            try:
                # Call the query service with progress callback
                result = await fastapi_app.state.query_service.process_query(
                    query_text, progress_callback=advance_progress
                )

                # Check if client is still connected before updating UI
                if not is_connected():
                    logger.info(
                        "Client disconnected during query processing, skipping UI update"
                    )
                    return

                # Clear loading
                try:
                    results_container.clear()
                except RuntimeError:
                    # Client was deleted, silently ignore
                    return

                # Restore search container
                if search_container_ui:
                    search_container_ui.set_visibility(True)

                with results_container:
                    if result.success and result.answer:
                        # Success - show answer and sources
                        render_answer(result)
                        render_sources(result.passages)

                        # Persist result to localStorage for recovery if user navigates away
                        try:
                            import json

                            result_data = {
                                "answer": result.answer,
                                "execution_time": result.execution_time,
                                "finalists": len(result.finalists),
                                "passages": [
                                    {
                                        "content": (
                                            passage.node.get_content()
                                            if hasattr(passage.node, "get_content")
                                            else str(passage.node)
                                        ),
                                        "title": (
                                            passage.node.metadata.get(
                                                "paper_title", "Unknown"
                                            )
                                            if hasattr(passage.node, "metadata")
                                            else "Unknown"
                                        ),
                                        "section": (
                                            passage.node.metadata.get("section", "")
                                            if hasattr(passage.node, "metadata")
                                            else ""
                                        ),
                                        "url": (
                                            passage.node.metadata.get("pdf_url", "")
                                            or passage.node.metadata.get("url", "")
                                            if hasattr(passage.node, "metadata")
                                            else ""
                                        ),
                                    }
                                    for passage in result.passages
                                ],
                                "query": query_text,
                            }
                            ui.run_javascript(
                                f"localStorage.setItem('necthrall_last_result', JSON.stringify({json.dumps(result_data)}))"
                            )
                        except Exception as e:
                            logger.debug(
                                f"Failed to persist result to localStorage: {e}"
                            )
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
                    if search_container_ui:
                        search_container_ui.set_visibility(True)
                    with results_container:
                        render_exception_error(e)
                except RuntimeError:
                    # Client was deleted, silently ignore
                    pass

        # =====================================================================
        # HEADER
        # =====================================================================
        with ui.row().classes(
            "header-container items-center justify-between lg:justify-center w-full px-4 bg-white shadow-sm relative"
        ):
            # Logo and brand
            with ui.row().classes("items-center gap-2"):
                ui.image(LOGO_DIR).classes(
                    "w-12 h-12 md:w-20 md:h-20 object-contain"
                ).props("no-spinner")

                ui.label("Necthrall").classes(
                    "text-2xl md:text-4xl font-bold text-slate-800 leading-none pb-1 md:pb-2"
                )

            # About button
            ui.button("About", on_click=about_dialog.open).props("flat").classes(
                "text-slate-600 font-semibold "
                "lg:absolute lg:right-8 lg:top-1/2 lg:-translate-y-1/2"
            )

        # =====================================================================
        # MAIN CONTENT
        # =====================================================================
        with ui.column().classes("w-full items-center px-4 py-8"):
            with ui.column().classes("w-full items-center px-4"):

                # --- HERO SECTION START ---
                with ui.column().classes("items-center w-full text-center mb-4"):
                    # The Slogan / Main Title
                    ui.label("Science, Distilled.").classes(
                        "text-4xl md:text-5xl font-extrabold text-slate-800 drop-shadow-lg shadow-primary/50 mb-3 tracking-tight"
                    )
                    # The Subtitle (Value Prop)
                    ui.label(
                        "Ask complex questions. Get cited, verifiable answers."
                    ).classes(
                        "text-base md:text-lg text-slate-500 max-w-full md:max-w-2xl text-center md:text-left break-words"
                    )
                # --- HERO SECTION END ---
                # Search container
                with ui.row().classes(
                    "w-full max-w-2xl items-center bg-white rounded-full border border-slate-200 shadow-sm "
                    "pl-3 pr-1 py-1 gap-1 "
                    "md:pl-5 md:pr-2 md:py-1.5 md:gap-2 "
                    "focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 transition-all "
                    "no-wrap"
                ):
                    # 1. Search Icon
                    ui.icon("search").classes(
                        "text-slate-400 shrink-0 " "text-lg md:text-xl"
                    )

                    # 2. Input Field
                    search_input = (
                        ui.input(placeholder="What would you like to research today?")
                        .classes("flex-grow min-w-0")
                        .props(
                            "borderless dense "
                            "input-style='font-size: 16px; text-overflow: ellipsis;' "
                            "input-class='placeholder-slate-400'"
                        )
                    )

                    # 3. Search Button
                    ui.button("Search", on_click=handle_search).props(
                        "flat dense rounded color=primary"
                    ).classes(
                        "font-bold shrink-0 " "px-3 text-sm " "md:px-6 md:text-base"
                    )

                    # Bind Enter key
                    search_input.on("keydown.enter", handle_search)

                # 2. Example Queries Row (Centered Below)
                example_queries_row = ui.row().classes(
                    "example-queries gap-2 mt-2 flex-wrap justify-center items-center"
                )
                with example_queries_row:
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
                        ).props("flat dense no-caps").classes(
                            "example-btn text-slate-600 hover:text-primary text-sm bg-slate-50 rounded-full px-3"
                        )

            # Results container
            results_container = ui.column().classes("w-full max-w-4xl mt-2")

            # Attempt to restore cached result from localStorage on page load
            def restore_cached_result():
                """Restore result from localStorage if available."""
                try:
                    import json

                    # Retrieve cached result from localStorage
                    cached_json = ui.run_javascript(
                        "localStorage.getItem('necthrall_last_result')", timeout=5
                    )
                    if cached_json:
                        cached_data = json.loads(cached_json)
                        search_input.value = cached_data.get("query", "")

                        # Populate the results container
                        with results_container:
                            # Render answer
                            with ui.column().classes("answer-section w-full"):
                                with ui.row().classes("items-center gap-3 mb-4"):
                                    with ui.row().classes("gap-2"):
                                        ui.label(
                                            f"⏱️ {cached_data['execution_time']:.1f}s"
                                        ).classes("stats-badge")
                                        ui.label(
                                            f"📄 {cached_data['finalists']} papers analyzed"
                                        ).classes("stats-badge")
                                        ui.label(
                                            f"📝 {len(cached_data['passages'])} sources cited"
                                        ).classes("stats-badge")

                            ui.markdown(cached_data["answer"]).classes("answer-text")

                            # Render sources
                            if cached_data["passages"]:
                                with ui.column().classes("sources-section w-full"):
                                    ui.label("SOURCES CITED").classes("sources-title")

                                    for idx, passage in enumerate(
                                        cached_data["passages"]
                                    ):
                                        with (
                                            ui.expansion()
                                            .classes("source-card")
                                            .props("dense") as expansion
                                        ):
                                            with expansion.add_slot("header"):
                                                with ui.column().classes("w-full"):
                                                    ui.label(passage["title"]).classes(
                                                        "source-title"
                                                    )

                                                    meta_parts = []
                                                    if passage["section"]:
                                                        meta_parts.append(
                                                            passage["section"]
                                                        )
                                                    meta_parts.append(
                                                        f"Citation [{idx + 1}]"
                                                    )
                                                    ui.label(
                                                        " • ".join(meta_parts)
                                                    ).classes("source-meta")

                                                    if passage["url"]:
                                                        with ui.row().classes(
                                                            "items-center mt-2"
                                                        ):
                                                            with (
                                                                ui.element("a")
                                                                .props(
                                                                    f'href="{passage["url"]}" target="_blank"'
                                                                )
                                                                .classes(
                                                                    "source-link-btn"
                                                                )
                                                            ):
                                                                ui.label(
                                                                    "📄 Open Access PDF"
                                                                )
                                                    else:
                                                        with ui.row().classes(
                                                            "items-center mt-2"
                                                        ):
                                                            ui.html(
                                                                '<span class="source-link-badge">📄 PDF not available</span>',
                                                                sanitize=False,
                                                            )

                                            with ui.column().classes("passage-content"):
                                                ui.label(passage["content"]).classes(
                                                    "source-snippet"
                                                )

                        logger.info(
                            f"✅ Restored cached result for query: {cached_data['query']}"
                        )
                except Exception as e:
                    logger.debug(f"No cached result to restore: {e}")

            # Run restoration on page load
            ui.timer(0.5, restore_cached_result, once=True)

        # =====================================================================
        # FOOTER (centered)
        # =====================================================================
        with ui.footer(fixed=False).classes(
            "bg-white text-slate-500 border-t border-slate-200 q-py-xl"
        ):
            # Center everything on all sizes: branding and links stacked and centered
            with ui.column().classes("w-full items-center gap-2 text-center"):
                ui.label(
                    f"© {str(datetime.date.today().year)} Necthrall. Research responsibly."
                ).classes("font-bold")
                ui.label(
                    "AI can make mistakes. Always verify important citations."
                ).classes("italic text-sm break-words max-w-full px-4")

                with ui.row().classes(
                    "gap-3 items-center justify-center flex-wrap mt-1"
                ):
                    ui.label("Privacy Policy").classes(
                        "cursor-pointer hover:text-slate-800 text-sm"
                    ).on("click", privacy_dialog.open)

                    ui.label("|").classes("text-slate-300")

                    ui.label("Terms of Service").classes(
                        "cursor-pointer hover:text-slate-800 text-sm"
                    ).on("click", terms_dialog.open)

                    ui.label("|").classes("text-slate-300")

                    ui.link(
                        "GitHub",
                        "https://github.com/adamcseresznye/necthrall",
                    ).classes("text-slate-500 hover:text-slate-800 text-sm")
