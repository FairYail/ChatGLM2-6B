import openai

# 设置你的OpenAI API密钥
openai.api_key = "YOUR_API_KEY"


def generate_chat_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']


# 在这里输入用户的对话开头
user_input = "你好啊"
response = generate_chat_response(user_input)
print(response)
