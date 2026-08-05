FROM python:3.14-slim

WORKDIR /app

# Pin versions: streamlit 1.61's internal GZip middleware calls
# GZipResponder with the pre-1.0 signature. Installing the latest
# starlette (1.x) breaks it with "missing thread_minimum_size" → 500.
RUN pip install --no-cache-dir "streamlit==1.61.0" "requests==2.32.*" "starlette<1.0"

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
