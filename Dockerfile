FROM python:3.8.19

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .


CMD gunicorn --bind 0.0.0.0:8000 app:app
