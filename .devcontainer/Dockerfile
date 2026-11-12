FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Same version that generated api/poetry.lock and core_ml/poetry.lock (see their
# headers), so the devcontainer resolves exactly what CI and Docker do.
ENV POETRY_VERSION=2.4.1

# make is the project's single entry point (Makefile).
# --no-install-recommends (DS-0029): no suggested packages nobody uses.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client make \
    && rm -rf /var/lib/apt/lists/*

# DS-0002: the devcontainers base image ships the non-root "vscode" user.
# Installing Poetry as that user puts it in the PATH of the account VS Code
# actually connects as.
USER vscode

# The official installer keeps Poetry's own dependencies away from the
# project's environment, which a pip install would mix.
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/home/vscode/.local/bin:$PATH"
