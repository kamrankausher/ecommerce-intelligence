FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run setup (generates data + trains model)
RUN python setup.py

EXPOSE 8000 8501

# Default: start API (use docker-compose for both services)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
