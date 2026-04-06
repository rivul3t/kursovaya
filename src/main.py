import json
import random
from tasks import TASKS
from prompts import build_teacher_prompt

from utils import llm_call, log_step, format_question

def generate_task(task_id, sample_index=0, topic=None, factor=None, difficulty=None, style=None, model="gpt-4o"):
    task = TASKS[task_id]
    
    topic = topic if topic else random.choice(task["topics"])
    style = style if style else random.choice(task["style"])
    factor = factor if factor else (random.choice(task["factors"]) if task.get("factors") else None)
    difficulty = difficulty if difficulty else "medium"

    prompt = build_teacher_prompt(task_id, topic, style, factor, difficulty, example=task.get("example"))
    
    log_step(task_id, sample_index, phase="prompt_generation", agent=model, action="generate_prompt", input_content=None, output_content=prompt)

    generated_text = llm_call(prompt, model)

    log_step(task_id, sample_index, phase="model_response", agent=model, action="generate_task", input_content=prompt, output_content=generated_text)

    try:
        generated_output = json.loads(generated_text)
    except Exception:
        generated_output = {"raw_text": generated_text}

    return generated_output

def solve_task(question_output, task_id, sample_index=0, model="gigachat"):
    question = format_question(question_output, task_id)
    prompt = f"""
    Дай ответ на задачу ниже.

    Правила:
    1. Сначала пошагово объясни ход мыслей (reasoning).
    2. Затем дай окончательный ответ (answer).
    3. Верни строго **только JSON** в следующем формате, без дополнительных комментариев, Markdown или текста:

    {{
      "reasoning": "пошаговое объяснение решения, каждый шаг через точку с запятой или отдельное предложение",
      "answer": "окончательный ответ на задачу"
    }}

    Вопрос:
    {question}

    ВАЖНО: JSON должен быть корректным и валидным. Никаких лишних символов, заголовков или пояснений.
    """
    
    log_step(task_id, sample_index, "student_prompt", "orchestrator", "build_student_prompt", input_content=prompt)

    raw_output = llm_call(prompt, model)

    try:
        answer_output = json.loads(raw_output)
    except Exception:
        answer_output = {"error": "invalid JSON", "raw_text": raw_output}

    answer_text = "\n".join(answer_output.get("context", [raw_output]))
    answer_output["text_output"] = answer_text

    log_step(task_id, sample_index, "student_response", model, "generate_answer", input_content=prompt, output_content=answer_output)
    return answer_output

def evaluate_task(question_output, answer_output, task_id, sample_index=0, model="gpt-4o"):
    question_output = format_question(question_output, task_id)

    prompt = f"Оцени корректность и качество ответа на вопрос:\nВопрос:\n{question_output}\nОтвет:\n{answer_output['text_output']}"
    log_step(task_id, sample_index, "evaluation_prompt", "orchestrator", "build_eval_prompt", input_content=prompt)

    raw_eval = llm_call(prompt, model)

    try:
        eval_result = json.loads(raw_eval)
    except Exception:
        eval_result = {"error": "invalid JSON", "raw_text": raw_eval}

    log_step(task_id, sample_index, "evaluation", model, "evaluate_output", input_content=prompt, output_content=eval_result)
    return eval_result

if __name__ == "__main__":
    task_id = "T1"  
    sample_index = 0
    #hardcode = "{\n  \"context\": [\n    \"Когнитивные искажения влияют на восприятие информации, заставляя людей переоценивать редкие события.\",\n    \"Эксперимент Стэнли Милгрэма продемонстрировал, как авторитетные фигуры могут усиливать подчинение даже против моральных убеждений.\",\n    \"Эффект плацебо показывает, что ожидания пациента могут изменять физиологические реакции независимо от активного лечения.\",\n    \"В рамках теории привязанности Джон Боулби утверждал, что безопасная привязанность в детстве способствует более стабильным межличностным отношениям во взрослом возрасте.\",\n    \"Нейронные сети в мозге функционируют точно так же, как процесс фотосинтеза у растений, преобразуя световую энергию в электрические сигналы.\"\n  ],\n  \"anomaly_index\": 4,\n  \"meta\": {\n    \"source\": \"GRE\",\n    \"topic\": \"психология\",\n    \"anomaly_type\": \"семантическое отклонение\"\n  }\n}"
    #question = json.loads(hardcode)

    question = generate_task(task_id, sample_index, difficulty='medium', model="openai/gpt-oss-120b")
    student_answer = solve_task(question, task_id, sample_index, model="openai/gpt-oss-120b")
    eval_result = evaluate_task(question, student_answer, task_id, sample_index, model="openai/gpt-oss-120b")

    print("=== FINAL EVAL RESULT ===")
    print(json.dumps(eval_result, ensure_ascii=False, indent=2))