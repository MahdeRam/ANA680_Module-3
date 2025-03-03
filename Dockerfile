FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY wine_model.pkl wine_model.pkl
COPY inference.py inference.py

EXPOSE 5000

CMD ["python", "inference.py"]
