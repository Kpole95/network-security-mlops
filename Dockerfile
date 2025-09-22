FROM python:3.10-slim-bookworm

WORKDIR /app
COPY . /app

# Install AWS CLI and clean up to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python3", "app.py"]