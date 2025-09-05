# Use official PyTorch image as base (with CUDA if needed)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Copy requirements and install
COPY backend/requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy backend code only (no outputs folder)
COPY backend/ .

# Expose port for API
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
