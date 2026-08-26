"""Config shim for the ported MoneyPrinterTurbo modules under topic_video/.

The upstream MPT modules (llm.py, voice.py, material.py, video.py, ...) all
do ``from app.config import config`` and then read plain dict-like
attributes off it: ``config.app.get(...)``, ``config.ui.get(...)``,
``config.proxy``. Upstream's real config.py backs those with a TOML file on
disk plus a WebUI; OpenShorts has no such file and no WebUI for this
pipeline; requests are BYOK per job. This module reproduces just the
attribute surface those ported modules touch, backed by an in-process,
per-thread dict instead of a config file.

Concurrency: ``app`` MUST be threading.local()-backed, not a plain module
dict. voice.py and material.py read ``config.app.get(...)`` directly with no
per-call override parameter, so two concurrent topic-video jobs sharing one
process-wide dict would leak each other's OpenAI/Pexels keys. Each pipeline
run sets ``mpt_config.app`` exactly once, at the top of the function that
runs inside ``run_in_executor`` -- a dedicated worker thread per job -- so
concurrent jobs never see each other's config.
"""

import threading


class _ThreadLocalApp(threading.local):
    """threading.local subclass so every thread gets its own ``dict``
    without callers having to remember to initialize it themselves."""

    def __init__(self):
        # threading.local.__init__ runs once per thread (the first time
        # that thread touches this instance), so this really does give each
        # thread a fresh empty dict rather than sharing one.
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def update(self, *args, **kwargs):
        self.data.update(*args, **kwargs)

    def clear(self):
        self.data.clear()


app = _ThreadLocalApp()

# Subtitle layout defaults. topic_video/schema.py's VideoParams reads these
# at class-definition (i.e. module import) time via config.ui.get(...), so
# this module must be fully initialized -- in particular `ui` must already
# be a plain dict with these two keys -- before schema.py is imported
# anywhere in the process.
ui = {
    "subtitle_position": "bottom",
    "custom_position": 70.0,
}

# The ported modules pass this straight to `requests.get(..., proxies=...)`.
# v1 has no outbound proxy support; None makes requests use no proxy at all
# (requests treats an empty dict the same way, but material.py's
# `_redact_request_error` iterates `(config.proxy or {}).values()`, which
# tolerates either).
proxy = None

# Only referenced in llm.py's Ollama branch, which v1 never reaches (OpenAI
# only). Kept as a stub purely so that code path doesn't hit an
# AttributeError if it's ever exercised by mistake.
def get_default_ollama_base_url() -> str:
    return "http://localhost:11434/v1"


# material.py's get_api_key() interpolates this into its "key not set" error
# message. There is no real config.toml backing this shim, so the message
# just names what BYOK header/config value was missing.
config_file = "<none: topic_video is BYOK-only, no config.toml>"
