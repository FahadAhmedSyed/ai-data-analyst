# Product Requirements Document: AI Data Analyst (MVP)

- Version: 1.0 (draft for approval)
- Date: 2026-11-22
- Owner: Fahad
- Related: [ADR index](adr/README.md), [architecture](architecture.md), [security](security.md), [privacy](privacy.md)

## 1. Summary

A web application where people upload a CSV or Excel file and ask questions about it in plain English. The app turns each question into SQL, checks the SQL, runs it on the uploaded data, and answers with a plain-language explanation, a table and a chart. Every number in an answer comes from an executed query, never from the language model.

## 2. Problem

Many people who work with spreadsheets (analysts, researchers, consultants, students, small businesses) cannot write SQL or Python. Spreadsheet tools make simple totals easy but grouped, filtered or time-based questions slow and error-prone. General chatbots can talk about data but estimate or invent numbers.

## 3. Users

| Persona | Typical data | What they need |
| --- | --- | --- |
| Business analyst | Sales, finance, operations exports | Fast answers to grouped and time-based questions; SQL they can check |
| Researcher or scientist | Survey results, field measurements, sensor time series | Correct handling of missing values, dates and units |
| Consultant or small business owner | Client spreadsheets of varying quality | Works on messy files; clear about assumptions |
| Student | Course datasets | Learns by seeing the SQL behind each answer |
| Environmental analyst (future focus) | Rainfall, river flow, water quality | Correct seasons, censored values, totals versus averages |

## 4. Goals and non-goals

### Goals for the MVP

1. A non-technical user can upload a file, understand its contents and get correct answers to typical analytical questions without help.
2. Answers are verifiably correct: the SQL, row count and assumptions are always visible.
3. Unsafe or failing queries never produce a guessed answer.
4. Accuracy is measured on a benchmark and meets the thresholds in ADR 0013.

### Non-goals for the MVP

- Accounts, saved conversations, or any persistence beyond the browser session (Phase 5).
- Multiple files or joins between tables (Phase 3).
- Statistical tests, regression, forecasting, anomaly detection (Phase 3).
- Multi-agent workflows (Phase 4).
- Excel, PDF or chart export (Phase 2). CSV download of results is in scope.
- Editing or cleaning the uploaded data. The app reports quality issues but never changes the data.
- Domain-specific environmental logic beyond flagging censored values.

## 5. Key user journeys

### J1: Upload and understand a dataset

1. The user enters the shared password.
2. The user uploads `rainfall_2020_2024.xlsx`.
3. The app lists three sheets; the user picks `Daily`.
4. The app detects that row 3 holds the headers and that `Date` is day-first. It shows a preview, the column list with types, missing values per column, numeric ranges and the date range.
5. A quality summary notes 412 missing rainfall values and 6 duplicate rows.

### J2: Ask and verify

1. The user asks "What was the average rainfall at each site during summer?"
2. The app shows "Interpreted as: average daily rainfall per site, December to February, all years" and the assumption "Summer = Dec to Feb (NZ default)".
3. It shows the answer in a sentence, a table of sites and averages, and a bar chart.
4. The user expands "Show SQL" and sees the query and "6 rows returned".

### J3: Follow up

1. The user asks "What about winter?" and gets the same analysis for June to August.
2. The user asks "Compare Site A and Site B over the last 12 months" and sees the resolved range ("Jan to Dec 2024, the latest 12 months in the data") with a monthly line chart.

### J4: Ambiguity

1. The user asks "Which sites had high rainfall?"
2. The app asks: "What counts as high? For example, the top 10% of daily values (above 24.6 mm), or days above a threshold you choose."

### J5: Failure

1. The user asks about a column that doesn't exist ("average wind speed").
2. The app answers that the dataset has no wind data, lists the available measures, and suggests a rephrasing. No numbers are shown.

## 6. Functional requirements

Priority: Must = required for MVP completion; Should = MVP if time allows, otherwise Phase 2.

| ID | Requirement | Priority | Tasks |
| --- | --- | --- | --- |
| FR-1.1 | Upload CSV files up to the configured size limit | Must | T1.03, T1.18 |
| FR-1.2 | Upload `.xlsx` and `.xls` files and choose one worksheet | Must | T1.04, T1.18 |
| FR-1.3 | Reject unsupported, empty, oversized or corrupt files with a specific message | Must | T1.03, T1.04 |
| FR-1.4 | Let the user override the detected delimiter and header row | Should | T1.03, T1.04, T1.18 |
| FR-2.1 | Detect file encoding and delimiter | Must | T1.03 |
| FR-2.2 | Detect the header row, including title rows above it | Must | T1.04 |
| FR-2.3 | Infer column types: integer, decimal, text, boolean, date, timestamp | Must | T1.05 |
| FR-2.4 | Parse day-first dates by default (NZ) and flag columns where day and month are ambiguous | Must | T1.05 |
| FR-2.5 | Parse numbers stored as text with thousands separators, currency signs or percent signs | Must | T1.05 |
| FR-2.6 | Flag censored values such as `<0.5` or `ND` without converting them | Must | T1.05 |
| FR-2.7 | Map original column names to unique, safe SQL identifiers; always display original names | Must | T1.05 |
| FR-3.1 | Show row and column counts, column names and types | Must | T1.07, T1.18 |
| FR-3.2 | Show missing-value counts and percentages per column | Must | T1.07 |
| FR-3.3 | Show min, max, mean, standard deviation and quartiles for numeric columns | Must | T1.07 |
| FR-3.4 | Show distinct counts and most frequent values for text columns | Must | T1.07 |
| FR-3.5 | Show date and timestamp ranges | Must | T1.07 |
| FR-3.6 | Show a preview of the first rows | Must | T1.18 |
| FR-3.7 | Show a data quality summary: missing values, duplicate rows, mixed-type columns, ambiguous dates, censored values | Must | T1.07, T1.18 |
| FR-4.1 | Accept a question in plain English through a chat input | Must | T1.18 |
| FR-4.2 | Produce a structured intent and SQL using only the real schema | Must | T1.11, T1.12 |
| FR-4.3 | Validate every query before execution (ADR 0005) | Must | T1.08 |
| FR-4.4 | Retry fixable failures at most twice, revalidating each attempt | Must | T1.13 |
| FR-4.5 | Execute with timeout, memory cap and row cap | Must | T1.06, T1.09 |
| FR-5.1 | Ask a clarifying question when readings differ in result and no default applies (ADR 0011) | Must | T1.12 |
| FR-5.2 | Apply configured region defaults and state them as assumptions | Must | T1.11 |
| FR-5.3 | Anchor relative dates to the dataset's latest date and state the range (ADR 0009) | Must | T1.11 |
| FR-6.1 | Explain each result in plain language with units where known | Must | T1.14 |
| FR-6.2 | Ensure every number in the explanation appears in the result (ADR 0017) | Must | T1.14 |
| FR-6.3 | Say clearly when the result is empty, truncated, or affected by missing data | Must | T1.14 |
| FR-7.1 | Resolve follow-up questions using the last three turns | Must | T1.16, T1.12 |
| FR-7.2 | Show the resolved question above each answer | Must | T1.18 |
| FR-7.3 | Reset the conversation while keeping the dataset | Must | T1.16, T1.18 |
| FR-7.4 | Replace the dataset, clearing the conversation after confirmation | Must | T1.16, T1.18 |
| FR-8.1 | Show results as a table | Must | T1.18 |
| FR-8.2 | Recommend and draw line, bar, scatter or histogram charts, or none | Must | T1.15 |
| FR-8.3 | Download results as CSV | Must | T1.18 |
| FR-8.4 | Change a chart through a follow-up question ("make it monthly") | Should | T1.15, T1.17 |
| FR-9.1 | Show the SQL, run status, row count, filters and assumptions for every answer | Must | T1.17, T1.18 |
| FR-9.2 | Show progress while a question is being processed | Must | T1.18 |
| FR-10.1 | Gate the app with a shared password (ADR 0012) | Must | T1.18 |
| FR-10.2 | Limit each session to 20 questions per 10 minutes | Must | T1.18 |
| FR-10.3 | Show a privacy notice on upload and offer schema-only mode (ADR 0010) | Must | T1.18 |

## 7. Non-functional requirements

| ID | Area | Requirement | Verified by |
| --- | --- | --- | --- |
| NFR-1 | Correctness | No number shown to the user is produced by the LLM | Grounding tests, benchmark |
| NFR-2 | Accuracy | Benchmark thresholds in ADR 0013 | Benchmark report |
| NFR-3 | Security | No LLM-generated SQL runs without validation; DuckDB cannot reach files, network, extensions or settings | Adversarial suite, lockdown tests |
| NFR-4 | Isolation | No session can read another session's data | Two-session AppTest |
| NFR-5 | Privacy | Only the content defined in ADR 0010 is sent to OpenAI; uploads are never written to disk | Prompt snapshot tests, code review |
| NFR-6 | Performance | 50 MB file loads in under 20 s; 1 million-row profile in under 5 s; question p95 under 15 s | Timing tests, benchmark |
| NFR-7 | Cost | Median cost per question within the threshold set after spike T0.4 | Benchmark report |
| NFR-8 | Reliability | Every failure ends in a specific user message; no stack traces shown | Integration tests |
| NFR-9 | Maintainability | Type hints, ruff-clean, at least 80% line coverage on non-UI modules | CI |
| NFR-10 | Observability | Structured logs with timings, token use, cost and error categories; no dataset contents | Log tests |
| NFR-11 | Portability | Runs locally with one command and on Railway with the same code | Deployment smoke test |
| NFR-12 | Usability | A non-technical tester completes J1 to J3 on an unseen file without guidance | UI acceptance test |

## 8. Constraints and assumptions

- Hosting: Railway, single instance; a restart clears sessions.
- LLM provider: OpenAI (ADR 0004); model chosen by spike.
- Limits: ADR 0014.
- Region defaults: New Zealand (ADR 0009).
- Datasets are one table at a time in the MVP.
- Testers are a small trusted group sharing one password.

## 9. MVP definition of done

The MVP is complete when all of the following hold:

1. Every Must requirement in section 6 is implemented and covered by at least one test.
2. All offline tests pass in CI.
3. The 150-case benchmark, including the held-out set, meets every threshold in ADR 0013.
4. The app is deployed on Railway behind the password, and the deployment process is documented.
5. Fahad completes journeys J1 to J5 on a dataset the system has not seen.
6. README, architecture, security, privacy, testing and deployment documents match the code.

## 10. Open questions

- Q1: Which model? Decided by spike T0.4 (c).
- Q2: The median cost-per-question threshold. Set after the same spike.
- Q3: Railway plan and memory size. Set from spike T0.4 (d).
