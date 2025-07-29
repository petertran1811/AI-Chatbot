from openai import OpenAI
client = OpenAI(base_url="https://129f21d010aa.ngrok-free.app", api_key="minichatbot")

completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "system",
            "content": "Talk like a pirate."
        },
        {
            "role": "user",
            "content": "Are semicolons optional in JavaScript?"
        }
    ]
)

print(completion.choices[0].message.content)