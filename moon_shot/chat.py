import sys
from openai import OpenAI

# 请确保替换为你的 API 密钥
api_key = ""
client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

# 初始化对话上下文
system_message = {
    "role": "user",
    "content": "你是我的私人助手，你的底层原理来自于大模型，不能出现如何关于Kimi的字样"
}
user_message = {"role": "user", "content": "你好"}
messages = [system_message, user_message]

# 持续聊天循环
while True:
    try:
        # 用户输入
        user_input = input("我: ")
        if user_input.lower() == 'exit':
            print("退出聊天。")
            break

        # 将用户输入添加到消息列表
        messages.append({"role": "user", "content": user_input})

        # 发送新消息并获取响应
        response = client.chat.completions.create(
        model="moonshot-v1-32k",
            messages=messages,
            temperature=0.3,
            stream=True,
        )
        collected_messages = []
        # 收集并打印新消息 取消换行
        print("AI:",end='')
        for idx, chunk in enumerate(response):
            chunk_message = chunk.choices[0].delta
            if chunk_message:  # 确保内容不为空
                # 逐字打印响应
                for key, value in chunk_message:
                    if key == 'content' and value != None:
                        sys.stdout.write(value)
                        sys.stdout.flush()
        print()  # 打印完一个字符后换行
    except Exception as e:
        print("发生错误：", e)
        break