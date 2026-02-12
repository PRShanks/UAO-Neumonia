FROM python:3.10-slim

WORKDIR /home/src

RUN apt-get update && apt-get install -y \
    python3-tk \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .


CMD ["python", "-m", "pytest", "-q"]


