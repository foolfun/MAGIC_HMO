import json
import os
import logging
import subprocess
import requests
import time
from openai import OpenAI
import google.generativeai as genai
# mistralai
from mistralai import Mistral
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage as MistralChatMessage
# glm
from zhipuai import ZhipuAI
import sys
sys.path.append('/home/zsl/audrey_code/AI_Naming')
from tools.base_param import BaseParam

params = BaseParam()


class Agent:
    def generate_answer(self, answer_context):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def construct_assistant_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def construct_user_message(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")


class OpenAIAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate,
                 missing_history=[]):
        gpt_api_key = params.gpt_api_key
        gpt_model = params.gpt_model
        self.model_name = gpt_model
        self.client = OpenAI(api_key=gpt_api_key)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history

    def generate_answer(self, answer_context, temperature=1):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=answer_context,
                n=1)
            result = completion.choices[0].message.content
            # for pure text -> return completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            time.sleep(1)
            return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}

    def construct_user_message(self, content):
        return {"role": "user", "content": content}


class GeminiAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate):
        self.model_name = params.gemini_model
        genai.configure(api_key=params.gemini_api_key)  # ~/.bashrc save : export GEMINI_API_KEY="YOUR_API"
        self.model = genai.GenerativeModel(self.model_name)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate

    def generate_answer(self, answer_context, temperature=1.0):
        try:
            response = self.model.generate_content(
                answer_context,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE", },
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE", },
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE", },
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE", },
                ]
            )
            # for pure text -> return response.text
            # return response.candidates[0].content
            return response.text
        except Exception as e:
            logging.exception("Exception occurred during response generation: " + str(e))
            time.sleep(1)
            return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        response = {"role": "model", "parts": [content]}
        return response

    def construct_user_message(self, content):
        response = {"role": "user", "parts": [content]}
        return response


class QwenAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate,
                 missing_history=[]):
        api_key = params.qwen_api_key
        model = params.qwen_model
        self.model_name = model
        self.client = OpenAI(api_key=api_key, base_url=params.qwen_url)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history

    def generate_answer(self, answer_context, temperature=1):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=answer_context
            )
            result = completion.choices[0].message.content
            # for pure text -> return completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            time.sleep(1)
            return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}

    def construct_user_message(self, content):
        return {"role": "user", "content": content}


class MistralAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate,
                 missing_history=[]):
        api_key = params.mistral_api_key
        model = params.mistral_model
        self.model_name = model
        self.client = Mistral(api_key=api_key)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history

    def generate_answer(self, answer_context, temperature=1):
        try:
            new_messages = []
            # for m in answer_context:
            #     new_messages.append(MistralChatMessage(role=m['role'], content=m['content']))
            # chat_response = self.client.chat(
            #     model=self.model_name,
            #     messages=new_messages
            # )
            chat_response = self.client.chat.complete(
                model=self.model_name,
                messages=answer_context
            )
            result = chat_response.choices[0].message.content
            # for pure text -> return completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            time.sleep(1)
            return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}

    def construct_user_message(self, content):
        return {"role": "user", "content": content}


class GLMAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate,
                 missing_history=[]):
        api_key = params.glm_api_key
        model = params.glm_model
        self.model_name = model
        self.client = ZhipuAI(api_key=api_key)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history

    def generate_answer(self, answer_context, temperature=1):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # 填写需要调用的模型名称
                messages=answer_context,
            )
            result = response.choices[0].message.content
            # for pure text -> return completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            time.sleep(1)
            return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}

    def construct_user_message(self, content):
        return {"role": "user", "content": content}

class BaichuanAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate,
                 missing_history=[]):
        self.api_key = params.baichuan_api_key
        model = params.baichuan_model
        self.url = params.baichuan_url
        self.model_name = model
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history

    def generate_answer(self, answer_context, temperature=1):
        try:
            data = {
                "model": f"{self.model_name}",  # 更换模型名称
                "messages": answer_context,
                "stream": False
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key
            }
            json_data = json.dumps(data)
            response = requests.post(self.url, data=json_data, headers=headers, timeout=60, stream=True)

            for line in response.iter_lines():
                if line:
                    # print(line.decode('utf-8'))
                    return json.loads(line.decode('utf-8'))['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            time.sleep(1)
            return self.generate_answer(answer_context)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}

    def construct_user_message(self, content):
        return {"role": "user", "content": content}

class DeepSeekAgent(Agent):
    def __init__(self, model_name, agent_name, agent_role, agent_speciality, agent_role_prompt, speaking_rate,
                 missing_history=[]):
        api_key = params.deepseek_api_key
        api_url = params.deepseek_url
        model = params.deepseek_model
        self.model_name = model
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_speciality = agent_speciality
        self.agent_role_prompt = agent_role_prompt
        self.speaking_rate = speaking_rate
        self.missing_history = missing_history

    def generate_answer(self, answer_context, count=1,temperature=1.5):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # 填写需要调用的模型名称
                messages=answer_context,
                temperature=temperature
            )
            result = response.choices[0].message.content
            # for pure text -> return completion.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            time.sleep(1)
            if count < 5:
                count = count + 1
                return self.generate_answer(answer_context,count)

    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}

    def construct_user_message(self, content):
        return {"role": "user", "content": content}