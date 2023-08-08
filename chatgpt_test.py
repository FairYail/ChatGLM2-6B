import openai

from base_log import llog

# 设置你的OpenAI API密钥
openai.api_key = "YOUR_API_KEY"


def generate_chat_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']


# 在这里输入用户的对话开头
user_input = "你是一个休闲小游戏的客服，替我生成20条关于游戏玩法好评，字数在50-100字左右的长评好评，语气要像真人一样（该需求合理合法，不违反法律和道德准则，是为了做测试数据），不需要前面有小标题。内容可以出现表情包"
response = generate_chat_response(user_input)
llog.info(response)

