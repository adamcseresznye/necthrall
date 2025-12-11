"""HTML injection constants for UI initialization."""

from pathlib import Path

# Get the directory where this file is located
_CURRENT_DIR = Path(__file__).resolve().parent

# Load PostHog analytics script from file
with open(_CURRENT_DIR / "analytics.js", "r") as f:
    POSTHOG_SCRIPT = f.read()

# Buy Me A Coffee widget injection
BUY_ME_COFFEE_WIDGET = """
<script data-name="BMC-Widget" data-cfasync="false" 
    src="https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js" 
    data-id="acseresznye" 
    data-description="Support Necthrall development" 
    data-message="" 
    data-color="#FFDD00" 
    data-position="Right" 
    data-x_margin="18" 
    data-y_margin="18">
</script>
"""
