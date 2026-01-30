"""HTML injection constants for UI initialization."""

from pathlib import Path

# Get the directory where this file is located
_CURRENT_DIR = Path(__file__).resolve().parent

# Load PostHog analytics script from file
with open(_CURRENT_DIR / "analytics.js", "r") as f:
    POSTHOG_SCRIPT = f.read()

# Ko-fi widget injection
# NOTE: Ko-fi's floating-chat button is rendered inside a same-origin about:blank iframe.
# On some systems/browsers (especially with dark mode), the iframe's default background
# becomes black (due to `color-scheme: light dark`), causing a black rectangle behind the pill.
# We fix that by forcing the iframe document background to transparent after it renders.
KOFI_WIDGET = """
<script src='https://storage.ko-fi.com/cdn/scripts/overlay-widget.js'></script>
<script>
  kofiWidgetOverlay.draw('necthrall', {
    'type': 'floating-chat',
    'floating-chat.donateButton.text': 'Support Us',
    'floating-chat.donateButton.background-color': '#FFD700',
    'floating-chat.donateButton.text-color': '#000000'
  });

  (function fixKofiFloatingChatBackground() {
    const startedAt = Date.now();
    const interval = setInterval(() => {
      const iframes = document.querySelectorAll(
        'iframe.floatingchat-container, iframe.floatingchat-container-mobi'
      );

      iframes.forEach((iframe) => {
        try {
          iframe.style.background = 'transparent';
          iframe.style.backgroundColor = 'transparent';

          const doc = iframe.contentDocument;
          if (!doc) return;

          // Force light color-scheme and transparent backgrounds to avoid the black box
          doc.documentElement.style.colorScheme = 'light';
          doc.documentElement.style.background = 'transparent';
          doc.documentElement.style.backgroundColor = 'transparent';
          if (doc.body) {
            doc.body.style.background = 'transparent';
            doc.body.style.backgroundColor = 'transparent';
          }
        } catch (e) {
          // Ignore cross-origin or timing errors
        }
      });

      if (Date.now() - startedAt > 6000) {
        clearInterval(interval);
      }
    }, 200);
  })();
</script>
"""
