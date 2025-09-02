FROM python:3.11

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai4adapt /ai4adapt
WORKDIR /ai4adapt/security

CMD ["python", "3-using_PPO.py"]
#CMD ["sleep","10m"]
