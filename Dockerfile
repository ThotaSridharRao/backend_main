# Use a Python 3.9 base image
FROM python:3.9-slim

# Set environment variables
ENV POETRY_VERSION=1.7.1
ENV PATH="/root/.local/bin:$PATH"
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=10000
ENV PYTHONUNBUFFERED=1

# Install Poetry
RUN apt-get update && apt-get install -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN poetry install --no-root

# Run the Flask app
CMD ["poetry", "run", "flask", "--app", "main", "run"]
