# predictor_ml_data — обучение модели для Stock Predictor App

Репозиторий для загрузки данных с MOEX, обучения модели прогноза и экспорта в TFLite для использования во Flutter-приложении.

## Требования

- **Python 3.10–3.12** (TensorFlow пока не поддерживает 3.13+; при 3.14 установите, например: `brew install python@3.12`)
- Доступ в интернет (MOEX ISS API)

## Установка

```bash
# Используйте python3.12, если у вас по умолчанию 3.14+
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## MOEX API

Используется публичный ISS API Московской биржи (без ключа):

- **История торгов (акции):**  
  `https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{TICKER}.json?from=YYYY-MM-DD&till=YYYY-MM-DD&start=0&iss.meta=off&iss.json=extended`

Модуль `moex_client.py` повторяет логику Flutter-приложения: пагинация по 100 записей, поля `TRADEDATE`, `CLOSE`, `VOLUME`, `HIGH`, `LOW`, дедупликация по дате.

### Пример загрузки данных

```python
from moex_client import fetch_stock_history, fetch_last_n_days

# За период
rows = fetch_stock_history("SBER", "2024-01-01", "2024-12-31")

# Последние N дней
rows = fetch_last_n_days("SBER", days=365)
```

## Обучение и экспорт TFLite

Из **того же терминала**, где делали `source .venv/bin/activate`, либо явно через интерпретатор venv:

```bash
# По умолчанию: тикер SBER, последние 730 дней, модель → stock_model.tflite
.venv/bin/python train_stock_model.py
# или после: source .venv/bin/activate
python train_stock_model.py

# Свои параметры
.venv/bin/python train_stock_model.py --ticker GAZP --days 500 --epochs 30 --out models/stock_model.tflite

# Явный период
.venv/bin/python train_stock_model.py --ticker SBER --from 2023-01-01 --till 2024-12-31 --out stock_model.tflite
```

Модель: окно последних **30 дней** (close, return, volume_norm, weekday) → предсказание **дневной доходности** на следующий день. После обучения сохраняется файл `.tflite`.

## Перенос модели в Flutter-приложение

1. Скопируйте `stock_model.tflite` в проект:
   ```text
   flutter_predictor_app/assets/models/stock_model.tflite
   ```
2. В `pubspec.yaml` добавьте:
   ```yaml
   flutter:
     assets:
       - assets/models/stock_model.tflite
   ```
3. Подключите пакет `tflite_flutter` и загружайте модель в приложении (интерфейс ввода: `[seq_len, features]`, вывод: следующая дневная доходность).

Параметры модели в коде: `SEQ_LEN=30`, `FEATURES=4` (close_norm, return, volume_norm, weekday), один выход (return). Эти же значения нужно использовать при вызове модели во Flutter.

## Структура репозитория

```text
predictor_ml_data/
  moex_client.py       # Клиент MOEX ISS API
  train_stock_model.py  # Обучение + экспорт .tflite
  requirements.txt
  README.md
  .gitignore
```
