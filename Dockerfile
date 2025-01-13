FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY serve /app/serve
RUN pip install --no-cache-dir -r requirements.tx

COPY . .

ENV MODEL_PATH=/app/reviews.pkl
ENV VECTORIZER_PATH=/app/tfidf.pkl

EXPOSE 8080

CMD ["python", "app.py"]