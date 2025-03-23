FROM python:3.10-slim

WORKDIR /src
ENV PYTHONPATH=/src

RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /src/logs /src/models /cron

COPY ./src /src

COPY ./docker/cron /cron

COPY ./docker/entrypoint.sh /src/entrypoint.sh
RUN chmod +x /src/entrypoint.sh

RUN echo "* * * * * /bin/sh /src/sh.sh >> /src/logs/cron.log 2>&1" > /cron/cron \
    && crontab /cron/cron

COPY ./src/requirements.txt /src/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8010

ENTRYPOINT ["/src/entrypoint.sh"]
