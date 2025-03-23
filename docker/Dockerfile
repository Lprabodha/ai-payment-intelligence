FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /src
ENV PYTHONPATH=/src 

RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /src/requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./src /src

COPY ./docker/entrypoint.sh /entrypoint.sh
COPY ./docker/cron /cron

COPY ../src/.env /src/.env

RUN chmod +x /entrypoint.sh

RUN mkdir -p /src/data/models

RUN mkdir -p /src/model

RUN mkdir -p /src/logs

RUN echo "" >> /cron && crontab /cron

EXPOSE 8010

ENTRYPOINT ["/entrypoint.sh"]