FROM python:3.10-slim

RUN pip install --no-cache-dir \
    "numpy<2.0" \
    torch==2.0.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    transformers==4.35.2 \
    accelerate==0.20.3 \
    datasets==2.13.1

WORKDIR /app

COPY train_model_small.py predict_model_small.py IMDB_Dataset.csv ./

# Clean up any residual pip cache
RUN rm -rf /root/.cache/pip

# Run script
CMD ["python", "train_model_small.py"]
