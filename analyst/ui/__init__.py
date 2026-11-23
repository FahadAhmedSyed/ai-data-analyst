"""The Streamlit interface (T1.18, ADR 0001).

A thin layer: each module here draws something and passes what the user did to
``analyst.session``, which holds the state and makes the decisions, and through it to the
pipeline. Nothing in ``analyst/session`` or ``analyst/pipeline`` imports Streamlit.

Rules for this package (security.md, T8 and T12):

* No ``unsafe_allow_html`` anywhere.
* Text from the file or the model is escaped (``escape_markdown``) unless it is the cleaned
  explanation, so it cannot become a link, an image or a heading.
* Nothing user-specific is kept in module globals, ``st.cache_data`` or ``st.cache_resource``;
  a session is one ``AnalystSession`` in ``st.session_state``.
"""
