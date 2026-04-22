import requests
import json
import dashscope
# sparkai
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage as SparkChatMessage
# ernie
import qianfan
# mistralai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage as MistralChatMessage
# openai
from openai import OpenAI
# glm
from zhipuai import ZhipuAI
from utils.base_param import BaseParam

params = BaseParam()

system_info = {"role": "system", "content": "请用中文作答。"}


class Baichuan:
    def __init__(self):
        self.MODEL = params.baichuan_model
        # print(f'Using model: {params.baichuan_model}')
        self.url = params.baichuan_url
        self.api_key = params.baichuan_api_key

    def get_answer(self, input_text):
        data = {
            "model": f"{self.MODEL}",  # 更换模型名称
            "messages": [
                system_info,
                {"role": "user", "content": f"{input_text}"}
            ],
            "stream": False
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        json_data = json.dumps(data)
        response = requests.post(self.url, data=json_data, headers=headers, timeout=60, stream=True)
        res = ''
        if response.status_code == 200:
            # print("请求成功！")
            # print("请求成功，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
            for line in response.iter_lines():
                if line:
                    # print(line.decode('utf-8'))
                    res = json.loads(line.decode('utf-8'))['choices'][0]['message']['content']
        else:
            print("请求失败，状态码:", response.status_code)
            print("请求失败，body:", response.text)
            print("请求失败，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
        return res

    def get_answerByMessages(self, messages):
        messages.insert(0, system_info)
        data = {
            "model": f"{self.MODEL}",  # 更换模型名称
            "messages": messages,
            "stream": False
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }
        json_data = json.dumps(data)
        response = requests.post(self.url, data=json_data, headers=headers, timeout=60, stream=True)
        res = ''
        if response.status_code == 200:
            # print("请求成功！")
            # print("请求成功，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
            for line in response.iter_lines():
                if line:
                    # print(line.decode('utf-8'))
                    res = json.loads(line.decode('utf-8'))['choices'][0]['message']['content']
        else:
            print("请求失败。")
        return res


class Qwen:
    def __init__(self):
        dashscope.api_key = params.qwen_api_key
        self.api_key = params.qwen_api_key
        self.url = params.qwen_url
        self.model = params.qwen_model
        self.client = OpenAI(
            api_key=self.api_key,  # 替换成真实DashScope的API_KEY
            base_url=self.url,  # 填写DashScopebase_url
        )

    def get_answer(self, input_text):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                system_info,
                {'role': 'user', 'content': input_text}
            ]
        )
        return completion.choices[0].message.dict()['content']

    def get_answerByMessages(self, messages):
        messages.insert(0, system_info)
        completion =  self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return completion.choices[0].message.dict()['content']


class Spark:
    def __init__(self):
        self.url = params.spark_url
        self.app_id = params.spark_appid
        self.api_secret = params.spark_api_secret
        self.api_key = params.spark_api_key
        self.domain = params.spark_domain

    def get_answer(self, input_text):
        model = ChatSparkLLM(
            spark_api_url=self.url,
            spark_app_id=self.app_id,
            spark_api_key=self.api_key,
            spark_api_secret=self.api_secret,
            spark_llm_domain=self.domain,
            streaming=False,
        )
        # temperature = 0.9,
        # max_tokens = 4096
        messages = [SparkChatMessage(role="system", content='你是一个中文智能助手'),
                    SparkChatMessage(role="user", content=input_text)]
        handler = ChunkPrintHandler()
        a = model.generate([messages], callbacks=[handler])

        return a.generations[0][0].text

    def get_answerByMessages(self, messages):
        messages.insert(0, system_info)
        new_messages = []
        for m in messages:
            new_messages.append(SparkChatMessage(role=m['role'], content=m['content']))
        model = ChatSparkLLM(
            spark_api_url=self.url,
            spark_app_id=self.app_id,
            spark_api_key=self.api_key,
            spark_api_secret=self.api_secret,
            spark_llm_domain=self.domain,
            streaming=False,
        )
        handler = ChunkPrintHandler()
        a = model.generate([new_messages], callbacks=[handler])
        return a.generations[0][0].text


class Ernie:
    def __init__(self):
        self.api_key = params.ernie_api_key
        self.secret_key = params.ernie_secret_key
        self.model_name = params.ernie_model

    def get_answer(self, input_text):
        chat_comp = qianfan.ChatCompletion(ak=self.api_key, sk=self.secret_key)
        # 指定特定模型
        resp = chat_comp.do(model=self.model_name,
                            messages=[
                                system_info,
                                {"role": "user", "content": input_text}
                            ])
        return resp["body"]

    def get_answerByMessages(self, messages):
        messages.insert(0, system_info)
        chat_comp = qianfan.ChatCompletion(ak=self.api_key, sk=self.secret_key)
        # 指定特定模型
        resp = chat_comp.do(model=self.model_name,
                            messages=messages)
        return resp["body"]


class GLM:
    def __init__(self):
        self.api_key = params.glm_api_key
        self.model_name = params.glm_model
        self.client = ZhipuAI(api_key=self.api_key)  # 填写您自己的APIKey
    def get_answer(self, input_text):
        response = self.client.chat.completions.create(
            model=self.model_name,  # 填写需要调用的模型名称
            messages=[
                system_info,
                {"role": "user", "content": input_text}
            ],
        )
        return response.choices[0].message.content

    def get_answerByMessages(self, messages):
        messages.insert(0, system_info)
        response = self.client.chat.completions.create(
            model=self.model_name,  # 填写需要调用的模型名称
            messages=messages,
        )
        return response.choices[0].message.content


class Kimi:
    def __init__(self):
        self.api_key = params.kimi_api_key
        self.model_name = params.kimi_model
        self.url = params.kimi_url

    def get_answer(self, input_text):
        client = OpenAI(api_key=self.api_key, base_url=self.url)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                system_info,
                {"role": "user", "content": input_text}
            ],
            temperature=0.2
            # 把 temperature 参数设置为一个较小的值。这个参数表示模型的采样温度，取值范围是 [0, 1]。较高的值（如 0.7）将使输出更加随机和发散，而较低的值（如 0.2）将使输出更加集中、更有确定性。
        )
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def get_answerByMessages(self, messages):
        messages.insert(0, system_info)
        client = OpenAI(api_key=self.api_key, base_url=self.url)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2)
        return completion.choices[0].message.content


class Mistral:
    def __init__(self):
        self.api_key = params.mistral_api_key
        self.model_name = params.mistral_model
        self.client = MistralClient(api_key=self.api_key)
    def get_answer(self, input_text):
        chat_response = self.client.chat(
            model=self.model_name,
            messages=[MistralChatMessage(role="system", content='你是一个中文智能助手。'),
                      MistralChatMessage(role="user", content=input_text)]
        )
        return chat_response.choices[0].message.content

    def get_answerByMessages(self, messages):
        messages.insert(0, system_info)
        new_messages = []
        for m in messages:
            new_messages.append(MistralChatMessage(role=m['role'], content=m['content']))
        chat_response = self.client.chat(
            model=self.model_name,
            messages=new_messages
        )
        return chat_response.choices[0].message.content


import google.generativeai as genai
class Gemini:
    def __init__(self):
        self.api_key = params.gemini_api_key
        self.model_name = params.gemini_model

    def get_answer(self, input_text):
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(model_name=self.model_name)
        response = model.generate_content(input_text)
        return response.text

    def get_answerByMessages(self, messages):
        new_messages = []
        for m in messages:
            if m['role'] == 'assistant':
                new_messages.append({'role': 'model', 'parts': m['content']})
            else:
                new_messages.append({'role': m['role'], 'parts': m['content']})
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(model_name=self.model_name)
        response = model.generate_content(new_messages)
        return response.text


class GPT:
    def __init__(self):
        self.api_key = params.gpt_api_key
        self.model_name = params.gpt_model
        self.client = OpenAI(api_key=self.api_key)
    def get_answer(self, input_text):

        # client = OpenAI(
        #     # defaults to os.environ.get("OPENAI_API_KEY")
        #     api_key=self.api_key,
        #     base_url="https://api.chatanywhere.tech/v1"
        #     # base_url="https://api.chatanywhere.cn/v1"
        # )

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[system_info,
                      {'role': 'user', 'content': input_text}]
        )
        return completion.choices[0].message.content

    def get_answerByMessages(self, messages):
        messages.insert(0, system_info)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return completion.choices[0].message.content


def choose_llm_model(llm_model):
    if llm_model == 'baichuan':
        return Baichuan()
    elif llm_model == 'qwen':
        return Qwen()
    elif llm_model == 'spark':
        return Spark()
    elif llm_model == 'mistral':
        return Mistral()
    elif llm_model == 'gemini':
        return Gemini()
    elif llm_model == 'gpt':
        return GPT()
    elif llm_model == 'ernie':
        return Ernie()
    elif llm_model == 'glm4':
        return GLM()
    elif llm_model == 'kimi':
        return Kimi()
    else:
        return None


def get_llm_response(llm_model, input_text):
    llm = choose_llm_model(llm_model)
    if llm is None:
        return 'No such model'
    return llm.get_answer(input_text)


def get_llm_responseByMessages(llm_model, messages):
    llm = choose_llm_model(llm_model)
    if llm is None:
        return 'No such model'
    return llm.get_answerByMessages(messages)


if __name__ == '__main__':
    print('test...')
    '''
        llm_model: baichuan, qwen,  mistral, gemini, gpt, glm4, kimi
    '''
    model = 'kimi'
    res = get_llm_response(model, '你好')
    print(res)

    # 以下为各个模型的测试代码

    # q = Qwen()
    # query = '这天有什么特别的地方？'
    # chat_li = [
    #     {"role": "system", "content": "假设你是一个对节日文化非常了解的专家。"},
    #     {"role": "user", "content": "你好,我想知道2024年6月1日是什么节日？"},
    #     {"role": "assistant", "content": "2024年6月1日是儿童节"},
    #     {"role": "user", "content": query}
    # ]
    # res = q.get_answerByMessages(messages=chat_li)
    # print(res)

    # s = Spark()
    # print(s.get_answer('你好'))

    # g = Gemini()
    # print(g.get_answer('你好'))

    # gpt = GPT()
    # print(gpt.get_answer('你好'))

    # ernie = ERNIE()
    # print(ernie.get_answer('你好'))

    # glm = GLM()
    # print(glm.get_answer('你好'))
