from gigachat import GigaChat
from config import GIGACHAT_API_TOKEN
from tools import find_nodes_in_city_tool

SYSTEM_PROMPT = """
Ты извлекаешь признаки местности из текста пользователя.

Если нужно искать объекты — вызывай tool find_nodes_in_city.

Передавай:

tags — ТОЛЬКО список OSM тегов БЕЗ значений.

Пример:

{"city": "Казань", "tags": ["bridge", "river"]}

НЕ передавай:
- значения
- русские слова
- словари
"""

async def quert_gigachat(user_text: str, city="Amsterdam"):
    with GigaChat(credentials=GIGACHAT_API_TOKEN, verify_ssl_certs=False) as gigachat:
        llm_response = gigachat.chat({
            "messages": [
                {"role": "user", "content": user_text}
            ],
            "functions": [find_nodes_in_city_tool],
            "function_call": "auto"
        })

    return llm_response