import os
from groq import Groq


text = """
Philosophers have long debated whether the universe has a beginning in time. Some argue that it must have started at some point, while others insist that it extends infinitely. 
Similarly, every complex object seems to be composed of simpler parts, yet some philosophers maintain that nothing is truly simple, and divisibility is infinite. 
The question of freedom also arises: humans seem to act freely, yet all events might be determined by natural laws. 
Finally, some claim that there must be a necessary being as the ultimate cause of everything, while others deny the existence of any absolutely necessary being.
"""


prompt = f"""
You are an expert in philosophical and logical analysis.

Analyze the following text carefully. Your task is to:
1. Identify statements that are in logical conflict with each other.
2. Explain why they contradict.
3. Output STRICTLY a JSON in this format:

{{
  "conflicts": [
    {{
      "statements": ["statement 1", "statement 2"],
      "explanation": "brief logical explanation of the conflict"
    }}
  ]
}}

Text to analyze:
{text}
"""

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }

    ]

)

print(completion.choices[0].message.content)