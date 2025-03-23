FROM python:3.10-slim

WORKDIR /src
ENV PYTHONPATH=/src

RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

COPY ./src/ /src/
COPY ./src/.env /src/.env
COPY ./docker/cron /cron

RUN mkdir -p /src/model
RUN mkdir -p /src/logs

RUN touch /src/entrypoint.sh

RUN echo '#!/bin/sh' > /src/entrypoint.sh && \
    echo 'echo "🚀 Starting Cron and FastAPI..."' >> /src/entrypoint.sh && \
    echo 'cron &' >> /src/entrypoint.sh && \
    echo 'uvicorn api.main:app --host 0.0.0.0 --port 8010 --reload' >> /src/entrypoint.sh

RUN chmod +x /src/entrypoint.sh

EXPOSE 8010

# ENTRYPOINT ["/src/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]