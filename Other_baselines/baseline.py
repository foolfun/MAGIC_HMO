import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import time

from utils.LLMs import *
from utils.base_param import BaseParam, getResults

from prompts.promptsBaseline import BaselinePromptDesign

from Other_baselines.query2keyword import PromptorQ2Kw, Query2Keyword
from Other_baselines.hyde_main.src.hyde import PromptorHyDE, HyDE

import os
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="/mnt/disk1/zsl/cache_huggingface_hub/hub/models--lier007--xiaobu-embedding-v2/snapshots/ee0b4ecdf5eb449e8240f2e3de2e10eeae877691",
    show_progress=True)
imp_db = Chroma(
    persist_directory='/home/zsl/audrey_code/AI_name/AI_name/db_id_metadata_implication_xiaobu_latest',
    embedding_function=embeddings)
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_community.chat_models import ChatBaichuan
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models import ChatZhipuAI

os.environ["TAVILY_API_KEY"] = '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'
tools = [TavilySearchResults(max_results=1)]
react_prompt = hub.pull("hwchase17/react")

bp = BaselinePromptDesign()
params = BaseParam()

parser = argparse.ArgumentParser(description="Run the baselines.")
parser.add_argument("-b", "--backbone", required=True, help="The backbone model to use.")
parser.add_argument("-ml", "--method_li", default="['base', 'fewshot', 'CoT','TDB','query2keyword']", help="The methods need to run.")
parser.add_argument("-n", "--number", default=-1, help="The number of queries to run. ")
args = parser.parse_args()

# args.backbone = 'baichuan'
# args.number = -1
# args.method_li = "['base']"

model = args.backbone  # 'baichuan', 'qwen','mistral', 'gemini', 'glm-4-long', 'gpt4o'
num = int(args.number)
f_baseline = params.test_bl_os + f'0818/baseline_{model}.csv'
df_exit = pd.DataFrame()
if os.path.exists(f_baseline):
    df_exit = pd.read_csv(f_baseline)
else:
    # 构建结果表
    df_ = pd.DataFrame(columns=['query', 'name', 'exp', 'r_poem', 'backbone', 'method', 'up_w', 'output'])
    df_.to_csv(f_baseline, index=False, encoding='utf-8', header=True)


def choose_react_llm(model: str):
    if model == "gpt":
        return ChatOpenAI(model=params.gpt_model, api_key=params.gpt_api_key)
    if model == "gemini":
        return ChatGoogleGenerativeAI(model=params.gemini_model, api_key=params.gemini_api_key)
    if model == "mistral":
        return ChatMistralAI(model=params.mistral_model, api_key=params.mistral_api_key)
    if model == "baichuan":
        return ChatBaichuan(model=params.baichuan_model, api_key=params.baichuan_api_key)
    if model == "qwen":
        return ChatTongyi(model=params.qwen_model, api_key=params.qwen_api_key)
    if model == "glm":
        return ChatZhipuAI(model=params.glm_model, api_key=params.glm_api_key)
    raise ValueError(f"Unknown model: {model}")


def searchK(method, documents, vector_db):
    # 检索知识
    ids = method.searchByEmb(search_query=documents, vector_db=vector_db, top_k=1)
    # 读取检索内容
    df_poems = pd.read_csv(params.data_file_os + 'poems_all_v3_06.csv', low_memory=False)
    df_final = df_poems[df_poems['id'].astype(str).isin(ids)]
    df_final = df_final.copy()
    df_final.loc[:, 'merge_info'] = df_final.apply(lambda row: (str(row['dynasty']) +
                                                                '·' + str(row['author'])
                                                                + '《' + str(row['title']) + '》' +
                                                                '\n诗句：' + str(row['content']) +
                                                                '。\n赏析：' + str(row['implication'])
                                                                ).replace('nan', ''), axis=1)
    poem = df_final.iloc[:]['merge_info'].values[0]
    return poem


def runHyDE(generator, query, prompt):
    promptor = PromptorHyDE('poem','zh')
    hyde = HyDE(promptor, generator)
    documents = hyde.generate(query)
    # print(documents)
    r_poem = searchK(hyde, documents, imp_db) # 检索查询仅llm生成的文档
    # 利用检索知识生成结果
    prom = prompt.format(user_query=query, poem=r_poem)
    res, output = getResults(llm=generator, prompt=prom, keys_=['名字', '解释'])
    return res, output, r_poem


def runQuery2Keywords(generator, query, prompt):
    promptor = PromptorQ2Kw('poem','zh')
    q2kw = Query2Keyword(promptor, generator)
    documents = q2kw.generate(query)
    # print(documents)
    documents = query + documents # 检索查询=用户需求+llm生成的文档
    r_poem = searchK(q2kw, documents, imp_db)
    # 利用检索知识生成结果
    prom = prompt.format(user_query=query, poem=r_poem)
    res, output = getResults(llm=generator, prompt=prom, keys_=['名字', '解释'])
    return res, output, r_poem

def saveToDf(llm, query_li, backbone, method, r_poem_li, up_w_li):
    prompt = ''
    print(f'Run {method}...')
    # 选择prompt
    if method == 'base':
        prompt = bp.prompt_base
    elif method == 'fewshot':
        prompt = bp.prompt_few_shot
    elif method == 'CoT':
        prompt = bp.prompt_CoT
    elif method == 'TDB':
        prompt = bp.prompt_TDB
    elif method == 'HyDE' or method == 'RAG' or method == 'query2keyword':
        prompt = bp.prompt_RAG
    # 读取query生成结果
    cnt = 0
    for q in tqdm(query_li):
        # 如果已存在，则跳过
        if df_exit.shape[0] > 0:
            df_res = df_exit[(df_exit['query'] == q)
                             & (df_exit['method'] == method)]
            if df_res.shape[0] > 0:
                cnt += 1
                continue
        # 生成prompt
        prom = ''
        if method == 'RAG':
            prom = prompt.format(user_query=q, poem=r_poem_li[cnt])
        elif method in ['base', 'fewshot', 'CoT', 'TDB']:
            prom = prompt.format(user_query=q)  # 生成基准模型的prompt
        try:
            # 获取结果
            r_poem = ''  # 初始化检索知识
            if method == 'HyDE':
                res, output, r_poem = runHyDE(generator=llm, query=q, prompt=prompt)
                if output == '':
                    print(f'Error in {method} for {q}')
                    continue
            elif method == 'query2keyword':
                res, output, r_poem = runQuery2Keywords(generator=llm, query=q, prompt=prompt)
                if output == '':
                    print(f'Error in {method} for {q}')
                    continue
            else:
                res, output = getResults(llm=llm, prompt=prom+'\n注意：返回JSON格式！！', keys_=['名字', '解释'])
                if output == '':
                    print(f'Error in {method} for {q}')
                    continue
        except:
            print(f'Error in {method} for {q}')
            continue
        if r_poem == '':
            r_poem = np.nan
        # 保存结果
        df_tmp = pd.DataFrame({'query': [q],
                               'name': [res['名字']],
                               'exp': [res['解释']],
                               'r_poem': [r_poem],
                               'backbone': [backbone],
                               'method': [method],
                               'up_w': [up_w_li[cnt]],
                               'output': [output]})
        df_tmp.to_csv(f_baseline, index=False, encoding='utf-8', mode='a', header=False)
        cnt += 1
    print(f'{method} Done!')


def choose_llm_run(backbone, query_li, r_poem_li, up_w_li):
    print(f'Use {backbone}')
    llm = None
    if backbone == 'baichuan':
        llm = Baichuan()
    elif backbone == 'spark':
        llm = Spark()
    elif backbone == 'qwen':
        llm = Qwen()
    elif backbone == 'ernie':
        llm = Ernie()
    elif backbone == 'glm4' or backbone == 'glm-4-long' or backbone == 'glm-4-flash':
        llm = GLM()
    elif backbone == 'gpt4o' or backbone == 'gpt4o_mini' or backbone == 'gpt4':
        llm = GPT()
    elif backbone == 'mistral':
        llm = Mistral()
    elif backbone == 'gemini':
        llm = Gemini()
    method_li = eval(args.method_li)  # TODO: 填入需要测评的方法
    for method in method_li:
        saveToDf(llm=llm, query_li=query_li, backbone=backbone, method=method, r_poem_li=r_poem_li, up_w_li=up_w_li)


if __name__ == '__main__':
    # 读取测试集
    df_data = pd.read_csv(params.f_test_dataset)
    if num == -1:
        num = df_data.shape[0]
    query_li = df_data['query'].values.tolist()[:num]
    r_poem_li = df_data['r_poem'].values.tolist()[:num]
    up_w_li = df_data['up_w'].values.tolist()[:num]
    # llm_names = ['baichuan', 'spark', 'qwen', 'ernie', 'glm-4-long', 'gpt4o', 'mistral', 'gemini']
    llm_names = [model]
    for bb in llm_names:
        print(f'Run {bb}')
        choose_llm_run(backbone=bb, query_li=query_li, r_poem_li=r_poem_li, up_w_li=up_w_li)
