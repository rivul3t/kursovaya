
import json
import re
import os
import requests
from groq import Groq
from gigachat import GigaChat
from typing import Dict, Any
import datetime
import copy
LOG_FILE = "process_logs.json"

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gigachat_client = GigaChat(credentials=os.environ.get("GIGACHAT_API_KEY"), verify_ssl_certs=False)

def gpt_call(prompt: str, model: str = "gpt-4o") -> str:
    try:
        res = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user","content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"gpt_call error: {e}")
        raise

def gigachat_call(prompt: str, model: str = "Gigachat") -> str:
    try:
        response = gigachat_client.chat(prompt)
        reply = response.choices[0].message.content.strip()
        return reply
    except Exception as e:
        print(f"[GIGACHAT CLIENT ERROR] {e}")
        raise

    except Exception as e:
        print(f"gigachat_call error: {e}")
        raise

def llm_call(prompt: str, model: str = "gpt-4o") -> str:
    if model.startswith("GigaChat"):
        return gigachat_call(prompt, model)
    else:
        return gpt_call(prompt, model)


# -- JSON extractor
def extract_json(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
        return json.loads(text)

    except (json.JSONDecodeError, TypeError) as e:
        print(f"🛑 JSON parsing error: {e}")
        print(f"🧪 Raw text (first 500 chars):\n{text[:500]}")
        raise

# -- Round-robin generator for topics/styles --
def round_robin(items):
    while True:
        for item in items:
            yield item

def format_question(example, task_id):
    if task_id in ["T1", "T2", "T5", "T6", "T7"]:
        return "\n".join(example["context"])
    elif task_id == "T3":
        return example["sentence"] + "\nВарианты: " + ", ".join(example["choices"])
    elif task_id == "T4":
        p1 = " ".join(example["paragraph_1"])
        p2 = " ".join(example["paragraph_2"])
        bridges = "\n".join([f"{i+1}. {b}" for i,b in enumerate(example["bridges"])])
        return f"{p1}\n{p2}\nМостовые предложения:\n{bridges}"
    else:
        return str(example)

process_logs = []

def log_step(task_id, sample_index, phase, agent, action, input_content=None, output_content=None, metadata=None, save_to_file=True):
    """
    Сохраняет лог для анализа и трассировки.
    
    Аргументы:
        task_id: ID задачи
        sample_index: индекс примера/сэмпла (например, счетчик цикла)
        phase: этап процесса (init, student_evaluation, difficulty_increase и т.д.)
        agent: участник процесса (teacher, student, orchestrator, system и т.д.)
        action: выполняемое действие (prompt, response, validate и т.д.)
        input_content: входные данные
        output_content: выходные данные
        metadata: дополнительная информация
    """

    timestamp = datetime.datetime.now().isoformat()

    # Создаем глубокую копию объектов, чтобы изменения позже не повлияли на лог
    if isinstance(input_content, dict):
        input_content = copy.deepcopy(input_content)
    if isinstance(output_content, dict):
        output_content = copy.deepcopy(output_content)
    if isinstance(metadata, dict):
        metadata = copy.deepcopy(metadata)
        
    log_entry = {
        "timestamp": timestamp,
        "task_id": task_id,
        "sample_index": sample_index,
        "phase": phase,
        "agent": agent,
        "action": action,
        "input": input_content,
        "output": output_content,
        "metadata": metadata or {}
    }
    process_logs.append(log_entry)

    if save_to_file:
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    old_logs = json.load(f)
            except Exception:
                old_logs = []
        else:
            old_logs = []

        old_logs.append(log_entry)

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(old_logs, f, ensure_ascii=False, indent=2)

def get_logs():
    """Возвращает все сохраненные логи"""
    return process_logs.copy()

def clear_logs():
    """Очищает хранилище логов"""
    process_logs.clear()