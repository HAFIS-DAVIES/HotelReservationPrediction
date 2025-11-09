# Use a lightweight Python image
FROM python:3.10-slim

# Environment vars for reliability
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Copy pre-trained model if local
COPY ./artifacts/models/lgbm_model.pkl /app/models/model.pkl

# Expose Flask port
ENV PORT=8080
EXPOSE 8080

# Start Flask app
CMD ["python", "application.py"]
