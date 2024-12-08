# Use the official Python 3.9 image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the necessary port
EXPOSE 8000

# Command to start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
