"""Page definitions for the Nechtrall UI."""

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
        ui.add_body_html(
            """
            <script data-name="BMC-Widget" data-cfasync="false" 
                src="https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js" 
                data-id="acseresznye" 
                data-description="Support Nechtrall development" 
                data-message="" 
                data-color="#FFDD00" 
                data-position="Right" 
                data-x_margin="18" 
                data-y_margin="18">
            </script>
            """
        )

        # Get client reference for connection checking
        client = context.client

        # =====================================================================
        # DIALOG: HOW IT WORKS
        # =====================================================================
        with ui.dialog() as about_dialog, ui.card().classes("q-pa-md"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("How Nechtrall Works").classes("text-xl font-bold")
                ui.button(icon="close", on_click=about_dialog.close).props(
                    "flat round dense"
                )

            with ui.column().classes("gap-4 mt-4"):
                # Step 1
                with ui.row().classes("items-center gap-4"):
                    ui.icon("travel_explore", size="md").classes("text-blue-500")
                    ui.label(
                        "1. Search: We scan millions of papers on Semantic Scholar."
                    ).classes("text-base")

                # Step 2
                with ui.row().classes("items-center gap-4"):
                    ui.icon("filter_alt", size="md").classes("text-green-500")
                    ui.label(
                        "2. Filter: We rank papers by citation count and impact."
                    ).classes("text-base")

                # Step 3
                with ui.row().classes("items-center gap-4"):
                    ui.icon("auto_awesome", size="md").classes("text-purple-500")
                    ui.label(
                        "3. Synthesize: AI reads the papers and writes a cited answer."
                    ).classes("text-base")

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
                ui.markdown(
                    """
                **Last Updated: December 2025**

                1. **No Personal Data Retention:** Nechtrall is a stateless application. We do not create user accounts, nor do we store your search history or personal details in our databases.
                
                2. **Third-Party Processing:** To answer your questions, your anonymized queries are processed by:
                   - **Semantic Scholar:** To retrieve relevant scientific papers.
                   - **LLM Providers (Google/Groq):** To analyze text and generate answers.
                   These providers may retain data transiently for abuse monitoring, but we do not store it.

                3. **Local Storage:** We use your browser's local storage solely to save session preferences (like dark mode), which never leave your device.
                """
                ).classes("text-slate-700")

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
                ui.markdown(
                    """
                **Last Updated: December 2025**

                1. **Research Purpose Only:** Nechtrall is an experimental AI tool designed for academic research assistance. It is **not** a substitute for professional medical, legal, or financial advice.

                2. **Accuracy Disclaimer:** AI models can make mistakes ("hallucinations"). While we strive to verify citations against real papers, **you must verify important claims** by reading the original source documents linked in the results.

                3. **No Liability:** The creators of Nechtrall accept no liability for any damages or losses resulting from reliance on the information provided by this tool. By using this service, you agree to cross-check all information.
                """
                ).classes("text-slate-700")

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

            # --- START: Enhanced Loading UI ---
            loading_timer = None
            with results_container:
                with ui.column().classes(
                    "w-full items-center justify-center py-12 gap-6"
                ):
                    # 1. Modern Ring Spinner
                    ui.spinner("rings", size="4rem", color="primary").props(
                        "thickness=4"
                    )

                    # 2. Cycling Status Message
                    status_label = ui.label("Initiating research scan...").classes(
                        "text-xl text-slate-600 font-medium animate-pulse text-center"
                    )

                    # 3. Tip
                    ui.label(
                        "This may take 15-30 seconds. We are reading real papers."
                    ).classes("text-sm text-slate-400 italic")

                # Simulated progress steps
                messages = [
                    "Querying Semantic Scholar API...",
                    "Filtering top candidate papers...",
                    "Downloading open-access PDFs...",
                    "Analyzing citation networks...",
                    "Extracting relevant passages...",
                    "Synthesizing final answer...",
                ]
                msg_idx = 0

                def update_progress():
                    nonlocal msg_idx
                    if msg_idx < len(messages):
                        status_label.set_text(messages[msg_idx])
                        msg_idx = (msg_idx + 1) % len(messages)

                # Update text every 2.5 seconds
                loading_timer = ui.timer(2.5, update_progress)
            # --- END: Enhanced Loading UI ---

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
            finally:
                # Clean up the timer to prevent errors
                if loading_timer:
                    loading_timer.cancel()

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

                ui.label("Nechtrall").classes(
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
            with ui.column().classes("w-full items-center px-4 py-8"):

                # --- HERO SECTION START ---
                with ui.column().classes("items-center w-full text-center mb-8"):
                    # The Slogan / Main Title
                    ui.label("Science, Distilled.").classes(
                        "text-4xl md:text-5xl font-extrabold text-slate-800 drop-shadow-lg shadow-primary/50 mb-3 tracking-tight"
                    )
                    # The Subtitle (Value Prop)
                    ui.label(
                        "Ask complex questions. Get cited, verifiable answers."
                    ).classes("text-lg text-slate-500 max-w-2xl")
            # --- HERO SECTION END ---
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

        # =====================================================================
        # FOOTER
        # =====================================================================
        with ui.footer(fixed=False).classes(
            "bg-white text-slate-500 border-t border-slate-200 q-py-xl"
        ):
            with ui.column().classes("w-full items-center gap-2"):
                ui.label("© 2025 Nechtrall AI. Research responsibly.").classes(
                    "font-bold"
                )
                ui.label(
                    "AI can make mistakes. Always verify important citations."
                ).classes("italic text-sm")

                with ui.row().classes("gap-4 items-center"):
                    ui.label("Privacy Policy").classes(
                        "cursor-pointer hover:text-slate-800 text-sm"
                    ).on("click", privacy_dialog.open)

                    ui.label("|").classes("text-slate-300")

                    ui.label("Terms of Service").classes(
                        "cursor-pointer hover:text-slate-800 text-sm"
                    ).on("click", terms_dialog.open)

                    ui.label("|").classes("text-slate-300")

                    ui.link(
                        "huggingface",
                        "https://huggingface.co/spaces/Necthrall/AI/tree/main",
                    ).classes("text-slate-500 hover:text-slate-800 text-sm")
