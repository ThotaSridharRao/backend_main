# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy Streamlit app
COPY . .

# Expose port and run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
