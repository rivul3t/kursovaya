from fastapi import FastAPI
from pydantic import BaseModel
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
    ReasoningEffort
)

app = FastAPI()

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, default="openai/gpt-oss-20b", help="HF model id")
parser.add_argument("--port", type=int, default=32666)
args = parser.parse_args()

model_id = args.model

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.2
    max_tokens: int = 4096


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):

    user_msg = req.messages[-1]["content"]

    system_message = (
        SystemContent.new()
        .with_model_identity("You are a helpful assistant")
        .with_reasoning_effort(ReasoningEffort.HIGH)
    )

    developer_message = (
        DeveloperContent.new()
        .with_instructions("Follow system instructions")
    )

    convo = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, user_msg),
    ])

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    inputs = {"input_ids": torch.tensor([tokens]).to(model.device)}

    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=True,
    )

    new_tokens = outputs[0][len(tokens):].tolist()

    parsed = encoding.parse_messages_from_completion_tokens(
        new_tokens,
        Role.ASSISTANT
    )

    final_text = ""
    for m in parsed:
        if m.channel == "final":
            final_text += m.content

    if not final_text and parsed:
        final_text = parsed[-1].content

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": final_text.strip()
                }
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)