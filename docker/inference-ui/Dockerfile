FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    torch==2.3.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    transformers==4.41.0 \
    uvicorn==0.29.0

# Copy source
COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]