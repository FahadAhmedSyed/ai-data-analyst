"""What one browser session holds and does, without any Streamlit (T1.18, architecture section 3).

The interface (``analyst/ui``) draws; this package decides. Keeping the two apart means the
rate limit, the loading of a file, the idle eviction and the shape of a conversation are
tested as plain Python, and the interface stays a thin layer over them (ADR 0001).
"""
