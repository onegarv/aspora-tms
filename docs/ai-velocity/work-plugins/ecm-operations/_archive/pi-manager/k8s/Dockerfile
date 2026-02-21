FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY queries/ ./queries/
COPY main.py .

# Run as non-root
RUN useradd -m appuser
USER appuser

ENTRYPOINT ["python", "main.py"]
CMD ["triage"]
