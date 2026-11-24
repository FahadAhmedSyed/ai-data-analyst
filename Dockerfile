# The image Railway builds and runs (T1.20; see docs/deployment.md).
#
# It installs exactly the packages in requirements.txt (exported from uv.lock, and checked
# against it in CI), copies only what the app needs at run time, and runs as a user that is not
# root. No secret is set here: the key and the password come from Railway variables.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # A container is a deployment: without APP_PASSWORD the app refuses to open (it does not
    # start open to the whole internet). Run it with -e APP_PASSWORD=... to try it locally.
    REQUIRE_GATE=true

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app.py ./
COPY .streamlit ./.streamlit
COPY analyst ./analyst

USER app

# Railway sets PORT; 8501 is the fallback for running the image on a laptop. One process, and
# one replica in Railway: every browser session lives in this process's memory.
EXPOSE 8501
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
