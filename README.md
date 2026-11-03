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
  `https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{TICKER}.json?from=YYYY-MM-DD&till=YYYY-MM-DD&start=0&iss.meta=off&iss.json=extended`

Модуль `moex_client.py` использует только основную площадку TQBR: пагинация
по 100 записей, поля `TRADEDATE`, `OPEN`, `CLOSE`, `VOLUME`, `HIGH`, `LOW`.

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

# Один тикер, свои параметры
.venv/bin/python train_stock_model.py --ticker GAZP --days 500 --epochs 30 --out models/stock_model.tflite

# Все ликвидные тикеры MOEX (площадка TQBR — основные торги; при недоступности API — встроенный список)
.venv/bin/python train_stock_model.py --tickers all --days 730 --epochs 20

# Несколько тикеров через запятую
.venv/bin/python train_stock_model.py --tickers SBER,GAZP,LKOH,YNDX --days 730

# Отдельная модель на каждый тикер (меньше нагрузка: обучается по одному тикеру за раз)
.venv/bin/python train_stock_model.py --tickers SBER,GAZP,LKOH --per-ticker --out models/
# → models/stock_model_SBER.tflite, models/stock_model_GAZP.tflite, ...

# Ограничить число тикеров при --tickers all (по умолчанию 500)
.venv/bin/python train_stock_model.py --tickers all --max-tickers 200 --days 365

# Явный период
.venv/bin/python train_stock_model.py --ticker SBER --from 2023-01-01 --till 2024-12-31 --out stock_model.tflite
```

Модель: окно последних **30 дней** (close, return, volume_norm, weekday) → предсказание **дневной доходности** на следующий день. После обучения сохраняется файл `.tflite`.

## Telegram-бот (MVP)

Бот принимает тикер MOEX, показывает прогноз на следующий торговый день и
отправляет график истории с прогнозной точкой.

1. Создайте бота через `@BotFather` и получите токен.
2. Установите зависимости и создайте конфигурацию:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

3. Запишите токен в `.env`:

```dotenv
TELEGRAM_BOT_TOKEN=ваш_токен
```

4. Обучите общую CatBoost-модель, если её ещё нет, и запустите бота:

```bash
python train_direction_model.py \
  --tickers all \
  --max-tickers 0 \
  --days 1500 \
  --volatility-threshold-multiplier 0 \
  --out models/stock_direction.cbm
python bot.py
```

Команды:

- `/predict SBER` — текстовый прогноз;
- `/chart SBER` — прогноз и PNG-график;
- `/ideas` — актуальные long-кандидаты на пять торговых сессий;
- сообщение `SBER` — текстовый прогноз.

Бот сначала ищет общую `models/stock_direction.cbm`, показывает вероятности
роста, боковика и падения и отсекает сигналы с уверенностью ниже 55%. Если
CatBoost-модели нет, используются модели TFLite. Явный путь можно задать
переменной `MODEL_PATH`.

Если присутствует `models/stock_signal_lgbm.txt`, бот дополнительно показывает
отдельное предупреждение следующей сессии. Стабильная policy допускает только
подтверждённый сигнал падения с вероятностью от 55% и отключается при рыночной
волатильности от 2%. В остальных случаях бот возвращает «нет подтверждённого
сигнала». CatBoost используется как резервная модель.

> Прогноз предназначен только для демонстрации и не является инвестиционной
> рекомендацией. Перед практическим использованием модель необходимо проверить
> на временном бэктесте.

## Проверка качества прогнозов

Walk-forward-бэктест обучает модель только на прошлом и проверяет на следующем
временном отрезке. Он сравнивает модель с нулевым прогнозом, последней дневной
доходностью и средней доходностью за 5 дней.

```bash
# Быстрая проверка одного тикера
.venv/bin/python backtest.py --tickers SBER --days 1500 --epochs 10

# Проверка 20 ликвидных тикеров
.venv/bin/python backtest.py \
  --tickers all \
  --max-tickers 20 \
  --days 1500 \
  --epochs 10 \
  --folds 3
```

В `reports/` сохраняются:

- JSON с итоговыми метриками по модели, baseline и каждому тикеру;
- CSV со всеми прогнозами и фактическими результатами;
- PNG с накопленной доходностью long/cash-стратегий.

Основные метрики: MAE, RMSE, точность направления, корреляция и доходность
после заданной комиссии (`--commission-bps`, по умолчанию 10 базисных пунктов).

Трёхклассовая CatBoost-модель проверяется отдельно:

```bash
.venv/bin/python backtest_direction.py \
  --tickers all \
  --max-tickers 0 \
  --days 1500 \
  --iterations 500 \
  --volatility-threshold-multiplier 0 \
  --folds 3
```

Цель модели: `down`, `flat` или `up` с порогом дневного изменения 0,3%.
Признаки включают доходности разных горизонтов, волатильность, SMA, RSI, ATR,
объём, диапазон дня, календарные признаки и динамику индекса IMOEX.
Для повторных запусков история кэшируется в `data/cache/`.
Зафиксированные результаты экспериментов приведены в
[`QUALITY_REPORT.md`](QUALITY_REPORT.md).

Бинарная модель торгового сигнала:

```bash
# Walk-forward-проверка
.venv/bin/python backtest_binary_signal.py \
  --tickers all \
  --max-tickers 0 \
  --days 1500 \
  --iterations 500

# Итоговое обучение
.venv/bin/python train_binary_signal_model.py \
  --tickers all \
  --max-tickers 0 \
  --days 1500 \
  --out models/stock_signal.cbm
```

Сравнение CatBoost, LightGBM и ансамбля:

```bash
.venv/bin/python backtest_ensemble_signal.py \
  --tickers all \
  --max-tickers 0 \
  --days 1500 \
  --iterations 500 \
  --commission-bps 10 \
  --slippage-bps 5

.venv/bin/python train_ensemble_signal_model.py \
  --tickers all \
  --max-tickers 0 \
  --days 1500
```

LightGBM сохраняется в `models/stock_signal_lgbm.txt`. Бот использует
зафиксированную stable warning-policy; CatBoost остаётся fallback.

Cross-sectional long ranker проверяется и обучается отдельно:

```bash
# Nested walk-forward для горизонтов 1/3/5 дней
.venv/bin/python backtest_ranker.py \
  --tickers all \
  --max-tickers 0 \
  --days 1500 \
  --horizons 1,3,5 \
  --commission-bps 10 \
  --slippage-bps 5

# Финальное обучение разрешено только для прошедшей gate H5/top 3% policy
.venv/bin/python train_ranker_model.py \
  --tickers all \
  --max-tickers 0 \
  --days 1500

# Обновлять после завершения торгов перед запуском/по расписанию
.venv/bin/python generate_long_ideas.py \
  --tickers all \
  --max-tickers 0 \
  --refresh
```

Команда бота `/ideas` читает `models/latest_long_ideas.json`. Ranker выбирает
только ликвидные бумаги, работает при положительном 20-дневном тренде IMOEX и
волатильности ниже 2%; иначе возвращает режим cash. Исторический H5 gate пройден,
но сигналы всё ещё требуют неизменяемой live-валидации.

Расширенный отчёт содержит precision/recall/coverage, log loss, Brier score,
calibration error, bootstrap 95% CI, результаты по fold, годам, режимам рынка
и ликвидности, а также автоматический quality gate. Модель не должна заменять
текущую, если gate не пройден.

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
  bot.py               # Telegram-бот
  predictor.py         # Инференс CatBoost/TFLite
  tabular_features.py  # Признаки CatBoost
  train_direction_model.py
  train_binary_signal_model.py
  train_ensemble_signal_model.py
  train_ranker_model.py
  backtest_direction.py
  backtest_binary_signal.py
  backtest_ensemble_signal.py
  backtest_ranker.py
  generate_long_ideas.py
  backtest.py
  moex_client.py       # Клиент MOEX ISS API
  train_stock_model.py  # Обучение + экспорт .tflite
  requirements.txt
  README.md
  .gitignore
```
