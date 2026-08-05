#!/usr/bin/env bash
# Точка входа воркера. Один образ — несколько ролей, роль задаётся командой.
set -euo pipefail

QUEUES="${CELERY_QUEUES:-cpu,io,gpu,net}"
CONCURRENCY="${CELERY_CONCURRENCY:-4}"
LOGLEVEL="${LOG_LEVEL:-INFO}"

# Потоки нативных библиотек — по числу ядер на процесс пула.
#
# OpenMP, BLAS и OpenCV читают эти переменные при загрузке и по умолчанию
# берут ВСЕ ядра — каждый процесс пула независимо от остальных. При
# concurrency=4 на 20 ядрах это 144 потока: процессор загружен на 1992 %,
# а работа стоит, потому что ядра заняты переключением контекста.
# Нагрузочный прогон Этапа 10 показал этап, который в одиночку идёт
# 17 секунд, а вчетвером не заканчивается десять минут.
#
# Выставляется здесь, а не в питоне: переменные действуют только до
# первого импорта numpy и torch.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  CORES="$(nproc 2>/dev/null || echo 4)"
  THREADS=$(( CORES / CONCURRENCY ))
  [ "${THREADS}" -lt 1 ] && THREADS=1
  export OMP_NUM_THREADS="${THREADS}"
  export MKL_NUM_THREADS="${THREADS}"
  export OPENBLAS_NUM_THREADS="${THREADS}"
  export NUMEXPR_NUM_THREADS="${THREADS}"
  echo "[entrypoint] ядер ${CORES}, процессов ${CONCURRENCY} → ${THREADS} потоков на процесс"
fi

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
