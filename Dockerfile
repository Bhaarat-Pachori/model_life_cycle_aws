FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=/app/reviews.pkl
ENV VECTORIZER_PATH=/app/tfidf.pkl

EXPOSE 8080

ENTRYPOINT [ "python", "app.py"]
# CMD [ "python", "app.py"]