"""HTML injection constants for UI initialization."""

from pathlib import Path

# Get the directory where this file is located
_CURRENT_DIR = Path(__file__).resolve().parent

# Load PostHog analytics script from file
with open(_CURRENT_DIR / "analytics.js", "r") as f:
    POSTHOG_SCRIPT = f.read()

# Ko-fi widget injection
# Uses your exact script + aggressive CSS to force a perfect Round Circle
KOFI_WIDGET = """
<script src='https://storage.ko-fi.com/cdn/scripts/overlay-widget.js'></script>
<script>
  kofiWidgetOverlay.draw('necthrall', {
    'type': 'floating-chat',
    'floating-chat.donateButton.text': 'Support Us',
    'floating-chat.donateButton.background-color': '#FFD700',
    'floating-chat.donateButton.text-color': '#000000'
  });
</script>
"""
