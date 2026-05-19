FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/streamlit ./frontend/streamlit

CMD ["streamlit", "run", "frontend/streamlit/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
