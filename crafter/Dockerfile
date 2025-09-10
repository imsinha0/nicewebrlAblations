FROM python:3.10-slim

RUN apt update && apt install -y curl procps git

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Explicitly copy things in .gitignore
#COPY ./config.py /app/

COPY ./google-cloud-key.json /app/

# Then copy everything else
COPY . /app

RUN uv sync --frozen


ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG

CMD ["uv", "run", "python", "web_app_assistant.py"]