FROM python:3.12-slim@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab
COPY --from=ghcr.io/astral-sh/uv:0.10.1 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local/ \
    PATH="/usr/local/bin:$PATH"

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev

COPY . .

EXPOSE 5001
CMD ["uv", "run", "gunicorn", "--workers", "1", "--bind", "0.0.0.0:5001", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "app:application"]
