# Privacy and data handling

- Version: 1.0 (MVP)
- Date: 2026-11-22
- Related: [ADR 0010](adr/0010-data-sent-to-openai.md), [security](security.md)

## 1. Summary for users

This is the text of the notice shown on upload (T1.18):

> Your file stays in this browser session's memory and is deleted when you close the tab or after 60 minutes of inactivity. To answer questions, the app sends OpenAI the column names and types, summary statistics, a few common values per column, three sample rows, and your question. Turn on schema-only mode to send no sample rows or values. OpenAI does not use API data to train its models by default.

## 2. What is stored, where, and for how long

| Data | Where | Retention |
| --- | --- | --- |
| Uploaded file bytes | Process memory only, during parsing | Discarded after loading |
| Dataset table | Session's in-memory DuckDB | Until the session ends or 60 idle minutes |
| Conversation | `st.session_state` | Same as above |
| Logs | Railway log stream | Railway's log retention; metadata only (see section 5) |
| Data sent to OpenAI | OpenAI | Per OpenAI terms (section 4) |

Nothing is written to disk by the app. A Railway restart deletes all sessions.

## 3. What is sent to OpenAI

| Content | Default mode | Schema-only mode |
| --- | --- | --- |
| Column names (original and SQL) and types | Sent | Sent |
| Row count, null rates, distinct counts | Sent | Sent |
| Numeric min and max, date ranges | Sent | Sent |
| Up to 20 most frequent values of low-cardinality text columns | Sent | Not sent |
| Three sample rows | Sent | Not sent |
| The question and the last three turns (questions, intents, SQL) | Sent | Sent |
| The first five rows of the most recent result, so follow-up questions can refer to it | Sent | Not sent (only its column names and row count) |
| Query results for explanation (up to 50 rows, or 20 rows plus summary statistics) | Sent | Sent (needed to explain the answer) |

All cell text is cut to 80 characters.

Schema-only mode still sends result rows to the explainer, because the answer must describe them. A stricter mode that explains results using only aggregate statistics is a possible Phase 2 addition if users need it.

## 4. OpenAI API terms (checked 2026-11-22)

From OpenAI's [Data controls in the OpenAI platform](https://developers.openai.com/api/docs/guides/your-data) page:

- Data sent to the API is not used to train OpenAI models unless the organisation opts in.
- Abuse-monitoring logs are kept for up to 30 days by default.
- The Responses API keeps application state for 30 days by default when storage is on. The LLM client sets `store=false` on every call, because the app does not need server-side state.
- Zero Data Retention is available only with OpenAI's prior approval.
- Regional data residency options exist ([data residency guide](https://developers.openai.com/api/docs/guides/data-residency)).

These terms change; they are re-checked before each deployment milestone. The secondary details above (application state, ZDR) came from a summary of the page, not a verbatim reading, and should be confirmed in OpenAI's dashboard settings before launch.

## 5. Logging

Logged: session hash, stage, timings, model, tokens, cost, validation and execution categories, row counts, SQL hash.

Never logged: file contents, sample rows, column values, result rows, answer text, the API key, the password. Full SQL text is logged only when `LOG_SQL_TEXT=true` for local debugging, because SQL can contain filter values from the data.

## 6. Recommendations before wider use

1. Use a dedicated OpenAI project for this app, with a spend cap.
2. Decide whether testers may upload confidential data. If not, say so in the notice.
3. If confidential data is in scope later, evaluate Zero Data Retention or a data residency option, and add a stricter privacy mode.
4. Add a privacy policy page before any public launch (Phase 5).
