FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default entrypoint (can be overridden by Kubeflow)
ENTRYPOINT ["python", "src/train_ddp.py"]
