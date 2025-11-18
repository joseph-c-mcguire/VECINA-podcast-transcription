FROM python:3.11-slim
WORKDIR /app
COPY . /app
# Install poetry
RUN pip install poetry
# Install dependencies
RUN poetry install --no-interaction --no-ansi
# Default command: run the module
ENTRYPOINT ["poetry", "run", "python", "-m", "vecina_transcriber"]
