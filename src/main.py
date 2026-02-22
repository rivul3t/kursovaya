import os
from groq import Groq


antinomy_statements = {
    2: "The world has a beginning in time and is limited in space",
    3: "Every composite substance consists of simple parts",
    1: "There exists an absolutely necessary being as the cause of the world",
    4: "Nothing composite consists of simple parts",
    7: "Freedom exists as a special kind of causality",
    5: "The world is infinite in time and space",
    6: "Freedom does not exist; everything is determined by laws of nature",
    8: "No absolutely necessary being exists"
}

gold_conflicts = {
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8)
}


prompt = """
You are an expert in logical reasoning and philosophical text analysis.

Step 1: Analyze the following statements carefully.
Step 2: Reason step by step, checking logical compatibility of each pair.
Step 3: Identify conflicts (contradictions) between statements.
Step 4: Output STRICTLY a JSON in this format:

{
  "conflicts": [
    {
      "statements": [id1, id2],
      "explanation": "brief logical explanation"
    }
  ]
}

Statements:
"""

for idx, text in antinomy_statements.items():
    prompt += f"{idx}. {text}\n"

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