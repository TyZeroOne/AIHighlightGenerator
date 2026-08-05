#!/usr/bin/env bash
# Точка входа воркера. Один образ — несколько ролей, роль задаётся командой.
set -euo pipefail

QUEUES="${CELERY_QUEUES:-cpu,io,gpu,net}"
CONCURRENCY="${CELERY_CONCURRENCY:-4}"
LOGLEVEL="${LOG_LEVEL:-INFO}"

case "${1:-worker}" in
  worker)
    # Имя ноды = <первая очередь>@<хост>, например "cpu@40b98ed97fb0".
    # Формула обязана совпадать с той, что использует HEALTHCHECK в Dockerfile,
    # иначе `celery inspect ping -d` не найдёт ноду и контейнер будет вечно
    # unhealthy при полностью живом воркере. Используется POSIX-подстановка
    # ${VAR%%,*}, потому что healthcheck выполняется в /bin/sh, а не в bash.
    NODE="${QUEUES%%,*}"
    echo "[entrypoint] celery worker | node=${NODE} queues=${QUEUES} concurrency=${CONCURRENCY}"
    exec celery -A workers.app worker \
      --queues="${QUEUES}" \
      --concurrency="${CONCURRENCY}" \
      --loglevel="${LOGLEVEL}" \
      --hostname="${NODE}@%h"
    ;;
  beat)
    echo "[entrypoint] celery beat"
    exec celery -A workers.app beat --loglevel="${LOGLEVEL}"
    ;;
  flower)
    exec celery -A workers.app flower --port=5555
    ;;
  shell)
    exec python
    ;;
  *)
    # Любая другая команда исполняется как есть — удобно для отладки:
    # docker compose run --rm worker-cpu python -c "..."
    exec "$@"
    ;;
esac
