# Hotpot contradiction dataset generator

## Run

```bash
pip install datasets openai groq
python -m gen_data.cli \
    --hf-split train \
    --output dataset_groq_50.jsonl \
    --backend groq \
    --model openai/gpt-oss-120b \
    --num-samples 50
```

For a quick mock test:

```bash
python -m gen_data.cli --hf-split train   --output dataset.jsonl   --backend mock
```
  
Avaible opitions
```bach
--input-json <json file>
--input-parquet <parquet file>
--hf-split <hugging face file split>

--output <filename>

--num-samples
--seed
--backend <mock/openai/groq>
--model <backend model name>

--none-ratio
--self-ratio
--pair-ratio
--conditional-ratio

--importance <most/least>
--length
--pair-position-cfg
--conditional-position-cfg

--max-retries
```