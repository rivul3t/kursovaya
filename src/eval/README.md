

## Запуск

```bash
python main.py \
  --input-jsonl data.jsonl \
  --output-json results.json \
  --backend openai \
  --model gpt-4.1-mini
```

---

## Параметры

### `--input-jsonl` (str, **required**)

Путь к входному файлу с сгенерированным датасетом.

Пример:

```bash
--input-jsonl data/hotpot.jsonl
```

---

### `--output-json` (str, **required**)

Путь, куда будут сохранены результаты.

Пример:

```bash
--output-json results/metrics.json
```

---

### `--backend` (str, optional)

Бэкенд для инференса модели.

Варианты:

* `mock` — заглушка
* `openai` — OpenAI API или совместимые сервисы
* `groq` — Groq API
* `gemini` — Google Gemini

По умолчанию:

```bash
mock
```

---

### `--base-url` (str, optional)

Base URL для API.

Используется для:

* OpenAI
* Ollama
* vLLM
* других совместимых сервисов

По умолчанию:

```bash
https://api.openai.com/v1
```

Примеры:

```bash
# OpenAI
--base-url https://api.openai.com/v1

# Ollama (локально)
--base-url http://localhost:11434/v1

# vLLM (локально)
--base-url http://localhost:8000/v1
```

---

### `--model` (str, optional)

Имя модели.

По умолчанию:

```bash
gpt-4.1-mini
```

Примеры:

```bash
--model gpt-4.1-mini
--model llama3
--model mistral
```

---

### `--prompt-style` (str, optional)

Стиль промпта.

Варианты:

* `basic` — простой prompt
* `cot` — Chain-of-Thought (рассуждение)

По умолчанию:

```bash
basic
```

---

## Примеры использования

### 🔹 OpenAI

```bash
python3 -m eval.cli \
  --input-jsonl data.jsonl \
  --output-json results.json \
  --backend openai \
  --model gpt-4.1-mini
```

---

### 🔹 Ollama

```bash
python3 -m eval.cli \
  --input-jsonl dataset_groq_5.jsonl \
  --output-json results_oss20b.json \
  --backend openai \
  --base-url http://localhost:32666/v1 \
  --model gpt-oss20b
```

---

### 🔹 vLLM

```bash
python3 -m eval.cli \
  --input-jsonl data.jsonl \
  --output-json results.json \
  --backend openai \
  --base-url http://localhost:8000/v1 \
  --model mistralai/Mistral-7B-Instruct-v0.2
```

---