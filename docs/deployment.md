# Deploying to Railway

- Version: 1.0 (T1.20)
- Date: 2026-11-24
- Related: [ADR 0019](adr/0019-container-deployment.md), [ADR 0012](adr/0012-alpha-access-control.md), [security](security.md), [privacy](privacy.md)

This guide is for the person who sets up the Railway project. Everything in the repository is ready; the steps below are the ones only the account owner can do. Nothing secret goes in the repository, in a chat or in a pull request: the key and the password are typed into Railway's Variables page and nowhere else.

## 1. What you end up with

One Railway service running one copy of the app, behind a shared password, at an address like `https://ai-data-analyst-production.up.railway.app`. Railway builds it from the `Dockerfile` in this repository each time `main` changes (or when you press Deploy).

A restart or a new deploy clears every session: people lose their loaded file and conversation and have to upload again. That is by design (ADR 0002), and the page says so.

## 2. Before you start

1. **An OpenAI project with a monthly spend cap.** The key belongs to this project only. The cap is what limits the cost if someone gets in (ADR 0012).
2. **A Railway plan that allows 2 GB of memory for one service.** Railway's pricing page (read on 2026-11-24, please check it again) lists 0.5 GB for the Free plan and 1 GB for the 30-day Trial, so neither is enough; **Hobby** (US$5 a month, 48 GB limit) is the smallest plan that fits. It also says an unverified Trial has restricted outbound network access, which would block the calls to OpenAI.
3. **The GitHub repository** `FahadAhmedSyed/ai-data-analyst`, which Railway reads.
4. **The four values for the variables** (section 4): the key, a password you choose, and the two model names.

## 3. Steps

1. Sign in to Railway and choose **New Project, Deploy from GitHub repo**, then `ai-data-analyst`. Allow Railway to read that repository.
2. Railway finds the `Dockerfile` and builds it. The first build takes a few minutes. It will fail to start the app properly until the variables are set; that is expected.
3. Open the service's **Variables** page and add the variables in section 4. Do not add `PORT`: Railway sets it.
4. In the service's **Settings**, check that:
   - the **health check path** is `/_stcore/health` (`railway.json` sets it; section 5 lists it if you need to type it in yourself);
   - the **number of replicas is 1**. The app keeps every session in the memory of one process, so two copies would send a person's next click to a copy that has never seen their file;
   - the **restart policy** is on failure, with up to 5 retries.
5. In **Settings, Networking**, generate a public domain. Railway gives it HTTPS.
6. Deploy. When the deploy shows as healthy, check it from any computer:

   ```bash
   python scripts/check_deployment.py https://your-address.up.railway.app
   ```

   It needs only Python, not the app's packages. It checks that the address is served over HTTPS, the health check says `ok`, the page loads, plain HTTP is sent to HTTPS, and no secret file is served.
7. **Open the address in a private browser window.** The password form must be the first and only thing on the page. Enter the password, load a small CSV, and ask one question. This is the one check the script cannot make, because Streamlit draws the page over a WebSocket.
8. Give the address and the password to the testers by a different route from each other (for example the address by email, the password by message).

## 4. Variables

Set these on the service's Variables page. The full list of settings, with defaults, is in [`.env.example`](../.env.example).

| Variable | Needed | Value |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | The key of the OpenAI project from section 2. Without it the page says questions are switched off |
| `APP_PASSWORD` | Yes | The shared password. **If it is missing or empty the app refuses to open** (see below) |
| `LLM_MODEL_GENERATE` | Yes | The model that writes the SQL, for example the one used in the benchmark (`gpt-4o-mini-2024-07-18`) |
| `LLM_MODEL_EXPLAIN` | Yes | The model that explains the result |
| `LLM_PRICES` | Recommended | JSON of prices per million tokens, so the logs show the cost of each question. The format is in `.env.example` |
| `MAX_UPLOAD_MB`, `MAX_ROWS`, `RATE_LIMIT_QUESTIONS`, `SESSION_IDLE_MINUTES` and the others | No | Only to change a limit from the default (ADR 0014) |
| `LOG_SQL_TEXT` | No | Leave unset. `true` writes SQL, which can contain values from the data, into the logs |

`REQUIRE_GATE=true` is set inside the image, so it is not a variable you add. It means a deployment never starts open: with no `APP_PASSWORD`, the page shows "This copy of the app has no password set, so it is closed" and nothing else, while the health check still answers, so Railway sees the service as running. Setting the password fixes it on the next deploy.

The upload limit is 50 MB in two places: `MAX_UPLOAD_MB` (the app) and `server.maxUploadSize` in `.streamlit/config.toml` (Streamlit's own server limit). Keep them equal. Raising it was not tested, and ADR 0014 sets 50 MB as the alpha limit.

## 5. The settings Railway needs (the same as `railway.json`)

| Setting | Value | Why |
| --- | --- | --- |
| Builder | Dockerfile (found automatically) | One image, the same on a laptop and on Railway |
| Start command | None: the image's `CMD` starts Streamlit on `$PORT`, on all addresses | Railway sets `PORT`; the app must listen there |
| Health check path | `/_stcore/health` | Streamlit's own route; it answers `ok` once the server is up. A path Streamlit does not serve, such as `/health`, makes Railway report the deploy as failed |
| Health check timeout | 60 seconds | The server is up a few seconds after the container starts |
| Restart policy | On failure, up to 5 retries | A crash or an out-of-memory stop brings the app back |
| Replicas | 1 | Sessions live in the process |

**A note on `railway.json`.** Railway's documentation (read on 2026-11-24) says Config as Code files such as `railway.json` are deprecated and keep working for existing services until **2026-12-01**; a forum answer says that after that date the file is no longer read. So the app does not depend on it: the start command is in the `Dockerfile`, which Railway always uses when it finds one, and `railway.json` carries only the health check and the restart policy. If the dashboard does not show those two values after the first deploy, type them in from the table above. Nothing else needs doing.

## 6. Memory

I started the image here with a 2 GB limit and measured it with a real browser (the numbers are Docker's, which are close to what Railway shows):

| State | Memory |
| --- | --- |
| Server started, nobody connected | about 65 MB |
| One browser session open (the app and its libraries loaded) | about 140 MB |
| After loading a 120,000-row, 5-column CSV (3.9 MB) in that session | about 220 MB |

So the server and its libraries cost about 140 MB, and a small session adds about 80 MB. The earlier measurement for a maximum-size file (about 150 MB steady and 300 MB at peak per session, [spikes](spikes.md)) still applies on top of the baseline, which is why the plan is **2 GB**: roughly 8 maximum-size sessions at once, or many more small ones. The app's own limits (ADR 0014) and the removal of idle sessions after 60 minutes are the protection if it gets busy. Files that are chosen but not yet loaded are held in memory too; they are capped at 8 maximum-size files (400 MB) over all sessions, so a visitor who has not signed in cannot use up the memory by sending files ([security](security.md) T13).

**The memory alert at 75%.** Railway's documentation (read on 2026-11-24) says monitors need the **Pro** plan (US$20 a month). If you are on Pro: open the project, then **Observability**, add a widget with the **Memory Usage** source, open its three-dot menu, choose **Add monitor**, and set the threshold to 1.5 GB (75% of 2 GB). Alerts go by email and in the app. It is a dashboard setting; it cannot be kept in the repository. On Hobby there is no alert: look at the service's **Metrics** graph now and then in the first weeks, and treat a line that keeps climbing as the sign to raise the limit or restart.

If the process runs out of memory, Railway restarts it (section 5) and every session is lost. The people affected see the "session expired" state and upload again.

## 7. Running the image on your own computer

To try exactly what Railway runs (needs Docker):

```bash
docker build -t ai-data-analyst .
docker run --rm -p 8501:8501 \
  -e APP_PASSWORD=choose-a-password \
  -e OPENAI_API_KEY=... -e LLM_MODEL_GENERATE=... -e LLM_MODEL_EXPLAIN=... \
  ai-data-analyst
```

Then open http://localhost:8501. Without `-e APP_PASSWORD=...` the page shows the closed message, which is the point. For everyday development use `uv run streamlit run app.py` instead ([README](../README.md)).

## 8. Operating it

| To do this | Do this |
| --- | --- |
| See what the app is doing | The service's **Logs**. They are JSON lines with the session (hashed), stage, duration, model, tokens, cost and error category, and never dataset values or answers ([privacy](privacy.md) section 5) |
| Change the password or the key | Edit the variable. Railway redeploys, which ends every session. Rotate the key at OpenAI too if it may have leaked |
| Roll back a bad deploy | Open the service's **Deployments** and redeploy an earlier one |
| Stop spending | Remove the service, or set the OpenAI project's cap lower. The cap is the real limit |
| Update the app | Merge to `main`; Railway builds and deploys it. Each deploy ends the sessions, so choose a quiet time |
| Update packages | Change `pyproject.toml`, run `uv lock`, then `uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt`. CI fails if the two files disagree, and the image installs `requirements.txt` |

## 9. When something is wrong

| What you see | Likely cause and fix |
| --- | --- |
| The deploy fails its health check | The health check path is not `/_stcore/health`, or the app crashed on start: read the logs. A start command typed into the dashboard that ignores `$PORT` does the same, so leave the start command empty |
| The page says "This copy of the app has no password set, so it is closed" | `APP_PASSWORD` is missing or empty. Add it |
| The page says questions are switched off | `OPENAI_API_KEY` or one of the two model names is missing |
| An answer fails with a connection error | The service cannot reach OpenAI: an unverified Trial plan, or a wrong key. Check the logs for the error category |
| An upload fails with "The server is holding too many uploads right now" | The 400 MB ceiling on files held in memory was reached, by busy testers or by someone sending files. It clears in a few minutes, when those sessions end. If it keeps happening, read the logs for `upload_refused` and check who is connected |
| Everyone was signed out and their files are gone | A restart or deploy. Check **Deployments** and the Metrics graph for an out-of-memory stop |
| The checker says a secret file is served | Stop: take the service offline, then find out which file and why. Rotate the key and the password |

## 10. What was and was not tested

Built and run here: the image builds from the `Dockerfile` and `requirements.txt`; the container runs as a user that is not root and answers `/_stcore/health` with `ok`; the password form, a wrong and a right password, loading a 120,000-row CSV, and the closed message with no password were driven in a real browser; the checker passes against the running container. Automated tests check the files and the checker (`tests/unit/test_deploy_files.py`, `tests/unit/test_check_deployment.py`) and the closed gate (`tests/e2e/test_app_gate.py`).

Not tested, because it needs the Railway account: the Railway build itself, the Railway health check, the public domain and HTTPS redirect, the dashboard settings, and the memory alert. The first deploy is the test of those; section 9 lists what usually goes wrong. The sandbox I built in could not reach PyPI from inside a container, so for that one build I installed the packages through the sandbox's proxy; every other line of the `Dockerfile` was built as it is in the repository.
