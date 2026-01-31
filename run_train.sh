#!/bin/bash
# Обучение модели на данных MOEX API.
# Нужен Python 3.10–3.12 (TensorFlow не поддерживает 3.14).
# Установка: brew install python@3.12

set -e
cd "$(dirname "$0")"

PYTHON=""
for p in python3.12 python3.11 python3.10 python3; do
  if $p -c "import sys; exit(0 if (3, 10) <= sys.version_info < (3, 13) else 1)" 2>/dev/null; then
    PYTHON=$p
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Нужен Python 3.10–3.12 (TensorFlow не поддерживает 3.13+)."
  echo "Установите: brew install python@3.12"
  exit 1
fi

# Пересоздать .venv, если он создан с Python 3.13+ (иначе пакеты не найдутся)
if [ -d ".venv" ]; then
  if ! .venv/bin/python -c "import sys; exit(0 if (3, 10) <= sys.version_info < (3, 13) else 1)" 2>/dev/null; then
    echo "Удаляю старый .venv (создан с Python 3.13+)..."
    rm -rf .venv
  fi
fi

if [ ! -d ".venv" ]; then
  echo "Создаю виртуальное окружение с $PYTHON..."
  $PYTHON -m venv .venv
fi
VENV_PYTHON=".venv/bin/python"
"$VENV_PYTHON" -m pip install -q -r requirements.txt

# TICKERS=all — обучить на всех тикерах MOEX; иначе один тикер (TICKER) или список через запятую
if [ -n "$TICKERS" ]; then
  TICKER_ARGS="--tickers $TICKERS"
else
  TICKER_ARGS="--ticker ${TICKER:-SBER}"
fi
echo "Загрузка данных с MOEX и обучение ($TICKER_ARGS, дни: ${DAYS:-730})..."
"$VENV_PYTHON" train_stock_model.py \
  $TICKER_ARGS \
  --days "${DAYS:-730}" \
  --epochs "${EPOCHS:-20}" \
  --out "${OUT:-stock_model.tflite}" \
  "$@"

echo "Готово. Файлы: stock_model.tflite, model_config.json"
