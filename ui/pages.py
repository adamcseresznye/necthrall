"""Page definitions for the Necthrall UI."""

import asyncio
import datetime
import time
from pathlib import Path

import httpx
from loguru import logger
from nicegui import app, context, ui

from config.config import get_settings
from ui.components import (
    SearchProgress,
    render_answer,
    render_error,
    render_exception_error,
    render_loading,
    render_sources,
)
from ui.constants import KOFI_WIDGET, POSTHOG_SCRIPT
from ui.policies import PRIVACY_POLICY, TERMS_OF_SERVICE

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
        # Inject Eva Icons CSS for the github icon
        ui.add_head_html(
            '<link href="https://unpkg.com/eva-icons@1.1.3/style/eva-icons.css" rel="stylesheet" />'
        )
        # Inject PostHog analytics script
        ui.add_head_html(f"<script>{POSTHOG_SCRIPT}</script>")

        # Inject custom CSS
        ui.add_head_html('<link rel="stylesheet" href="/static/css/styles.css">')

        # Inject Ko-fi widget
        ui.add_body_html(KOFI_WIDGET)

        # Get client reference for connection checking
        client = context.client

        state = {"is_searching": False, "last_search_time": 0}

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
                        "2. Filter: We rank results by semantic relevance, credibility, citation impact, and recency."
                    ).classes("text-sm md:text-base break-words flex-1")

                # Step 3: Highlight "Full Text" & "Inline Citations" (Your key V3 features)
                with ui.row().classes("items-start md:items-center gap-3"):
                    ui.icon("auto_awesome").classes(
                        "text-purple-500 shrink-0 text-xl md:text-2xl"
                    )
                    ui.label(
                        "3. AI reads the full PDFs, identifies specific relevant passages, and generates an answer with precise inline citations."
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

        # =====================================================================
        # DIALOG: CONTACT US
        # =====================================================================
        with (
            ui.dialog() as contact_dialog,
            ui.card().classes("w-full max-w-lg q-pa-lg"),
        ):
            with ui.row().classes("w-full items-center justify-between mb-4"):
                ui.label("Contact Us").classes("text-xl font-bold")
                ui.button(icon="close", on_click=contact_dialog.close).props(
                    "flat round dense"
                )

            with ui.column().classes("w-full gap-4"):
                c_name = ui.input("Name").classes("w-full").props("outlined dense")
                c_email = ui.input("Email").classes("w-full").props("outlined dense")
                c_msg = ui.textarea("Message").classes("w-full").props("outlined dense")

                async def handle_contact_submit():
                    # 1. Validate inputs (Python side)
                    if not c_name.value or not c_email.value or not c_msg.value:
                        ui.notify("Please fill out all fields", type="warning")
                        return

                    submit_btn.disable()

                    try:
                        if not get_settings().WEB3FORMS_ACCESS_KEY:
                            logger.warning("WEB3FORMS_ACCESS_KEY missing.")
                            ui.notify(
                                "Message logged (Service not configured).", type="info"
                            )
                        else:
                            # 2. Prepare Payload
                            payload = {
                                "access_key": get_settings().WEB3FORMS_ACCESS_KEY,
                                "name": c_name.value,
                                "email": c_email.value,
                                "message": c_msg.value,
                                "subject": f"Necthrall Contact: {c_name.value}",
                                "botcheck": False,
                            }

                            import json

                            # 3. Submit via Client-side JS (Web3Forms Free Tier requires this)
                            # Note: We use json.dumps to safely serialize the payload into the JS string.
                            js_code = f"""
                            const response = await fetch('https://api.web3forms.com/submit', {{
                                method: 'POST',
                                headers: {{
                                    'Content-Type': 'application/json',
                                    'Accept': 'application/json'
                                }},
                                body: JSON.stringify({json.dumps(payload)})
                            }});
                            return await response.json();
                            """

                            result = await ui.run_javascript(js_code, timeout=15.0)

                            if result and result.get("success"):
                                ui.notify(
                                    result.get("message", "Message sent!"),
                                    type="positive",
                                )
                                # Clear form only on success
                                contact_dialog.close()
                                c_name.value = ""
                                c_email.value = ""
                                c_msg.value = ""
                            else:
                                error_msg = (
                                    result.get("message", "Unknown error")
                                    if result
                                    else "No response"
                                )
                                logger.error(f"Web3Forms JS Error: {result}")
                                ui.notify(f"Failed: {error_msg}", type="negative")

                    except Exception as e:
                        logger.error(f"Contact form exception: {e}")
                        ui.notify(
                            "An error occurred. Please try again later.",
                            type="negative",
                        )
                    finally:
                        submit_btn.enable()

                submit_btn = (
                    ui.button("Send Message", on_click=handle_contact_submit)
                    .props("unelevated color=primary")
                    .classes("w-full font-bold")
                )

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
                return True

        async def handle_search():
            # Client Spam Check
            if state["is_searching"]:
                ui.notify("Search in progress", type="warning")
                return

            # Rate Limit Check
            if time.time() - state["last_search_time"] < 10:
                ui.notify(
                    "Please wait a moment before searching again.", type="warning"
                )
                return

            # Server Load Check
            if fastapi_app.state.search_semaphore.locked():
                limit = fastapi_app.state.max_concurrent_searches
                ui.notify(
                    f"Server is at capacity ({limit}/{limit}). Please try again later.",
                    type="warning",
                )
                return

            query_text = search_input.value.strip()

            if not query_text:
                ui.notify("Please enter a research question", type="warning")
                return
            # Basic validation: do not start pipeline for very short queries.
            # Require at least 10 characters.
            if len(query_text) < 10:
                ui.notify(
                    "Please provide a more specific query",
                    type="warning",
                )
                return

            # Update State
            state["is_searching"] = True
            state["last_search_time"] = time.time()

            # Clear previous results
            if results_container:
                try:
                    results_container.clear()
                except RuntimeError:
                    # Client deleted, exit gracefully
                    state["is_searching"] = False
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
                # Create a future to receive the result
                loop = asyncio.get_running_loop()
                future = loop.create_future()

                # Add to queue
                queue = fastapi_app.state.search_queue
                position = queue.qsize() + 1
                ui.notify(
                    f"You are number {position} in the queue. We'll start your search shortly!",
                    type="info",
                    position="top",
                    close_button="Got it",
                )

                await queue.put(
                    (
                        query_text,
                        deep_mode_switch.value,
                        advance_progress,
                        future,
                    )
                )

                # Wait for result
                result = await future

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

                            # Serialize passages (Lightweight for localStorage)
                            serialized_passages = []
                            for p in result.passages:
                                # Handle Pydantic Passage object
                                text = getattr(p, "text", "")
                                metadata = getattr(p, "metadata", {})
                                score = getattr(p, "score", None)

                                # Truncate content to prevent localStorage overflow
                                content_snippet = (
                                    text[:200] + "..." if len(text) > 200 else text
                                )

                                serialized_passages.append(
                                    {
                                        "content": content_snippet,
                                        "title": metadata.get("paper_title")
                                        or metadata.get("title", "Unknown"),
                                        "url": metadata.get("pdf_url")
                                        or metadata.get("url", ""),
                                        "score": score,
                                        "section": metadata.get("section", ""),
                                    }
                                )

                            result_data = {
                                "answer": result.answer,
                                "execution_time": result.execution_time,
                                "finalists": len(result.finalists),
                                "passages": serialized_passages,
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

            finally:
                state["is_searching"] = False

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
                    "text-2xl md:text-4xl font-bold text-slate-800 leading-none pb-1 md:pb-2 cursor-pointer"
                ).on(
                    "click",
                    lambda e=None: ui.run_javascript(
                        "if (window.location.pathname !== '/') { window.location.href = '/'; } else { window.location.reload(); }"
                    ),
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
                            "borderless dense autofocus"
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

                # Deep Search Switch (Moved below for mobile responsiveness)
                with ui.row().classes("w-full max-w-2xl justify-end px-2 -mt-1 mb-1"):
                    deep_mode_switch = (
                        ui.switch("Deep Search", value=False)
                        .props("dense color=primary")
                        .classes("text-slate-500 text-xs md:text-sm")
                    )
                    with deep_mode_switch:
                        ui.tooltip(
                            "Enable for deep analysis of full PDFs. Disable for faster search using abstracts only."
                        )

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
                                                    ui.label(
                                                        passage.get("title", "Unknown")
                                                    ).classes("source-title")

                                                    meta_parts = []
                                                    section = passage.get("section")
                                                    if section:
                                                        meta_parts.append(section)
                                                    meta_parts.append(
                                                        f"Citation [{idx + 1}]"
                                                    )
                                                    ui.label(
                                                        " • ".join(meta_parts)
                                                    ).classes("source-meta")

                                                    url = passage.get("url")
                                                    if url:
                                                        with ui.row().classes(
                                                            "items-center mt-2"
                                                        ):
                                                            with (
                                                                ui.element("a")
                                                                .props(
                                                                    f'href="{url}" target="_blank"'
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
                                                ui.label(
                                                    passage.get("content", "")
                                                ).classes("source-snippet")

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
                    f"© {str(datetime.date.today().year)} Necthrall. All rights reserved."
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

                    ui.label("Contact").classes(
                        "cursor-pointer hover:text-slate-800 text-sm"
                    ).on("click", contact_dialog.open)

                    ui.label("|").classes("text-slate-300")

                    ui.link(
                        "GitHub",
                        "https://github.com/adamcseresznye/necthrall",
                    ).classes("text-slate-500 hover:text-slate-800 text-sm").props(
                        'target="_blank"'
                    )
