
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY model/ ./model/
COPY app.py .

EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn api.main:app --host 127.0.0.1 --port 8000 & streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]