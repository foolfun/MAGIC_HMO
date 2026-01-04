from tools.LLMs import *
import pandas as pd
import sys
import os
import argparse
from tqdm import tqdm
from tools.base_param import BaseParam, getResults

params = BaseParam()
import numpy as np

sys.path.append('/home/zsl/audrey_code/AI_Naming/Other_baselines/hyde_main/src/')
from hyde import PromptorHyDE, OpenAIGenerator, CohereGenerator, HyDE


class Config:
    populate_by_name = True


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="/mnt/disk1/zsl/cache_huggingface_hub/hub/models--lier007--xiaobu-embedding-v2/snapshots/ee0b4ecdf5eb449e8240f2e3de2e10eeae877691",
    show_progress=True)
topics_db = Chroma(
    persist_directory='/home/zsl/audrey_code/AI_name/AI_name/db_id_metadata_topics_xiaobu_latest',
    embedding_function=embeddings)
imp_db = Chroma(
    persist_directory='/home/zsl/audrey_code/AI_name/AI_name/db_id_metadata_implication_xiaobu_latest',
    embedding_function=embeddings)

promptor = Promptor('poem')
generator = Qwen()

parser = argparse.ArgumentParser(description="Run the baselines.")
# parser.add_argument("-b", "--backbone", required=True, help="The backbone model to use.")
# parser.add_argument("-ml", "--method_li", default="['hyde']", help="The methods need to run.")
# parser.add_argument("-n", "--number", default=-1, help="The number of queries to run. ")
args = parser.parse_args()

args.backbone = 'glm-4-flash'
args.number = -1
args.method_li = ['hyde']

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

hyde = HyDE(promptor, generator)

df = pd.read_csv(params.test_bm_os + 'test_data_500.csv', low_memory=False)
for i in tqdm(range(0, 500)):
    query = df.loc[i, 'query']
    # prompt = hyde.prompt(query)
    # print(prompt)
    hypothesis_documents = hyde.generate(query)
    # print(hypothesis_documents)
    hits = hyde.searchByEmb(search_query=hypothesis_documents, vector_db=imp_db, top_k=1)
    # print(hits)
    df_poems = pd.read_csv(params.data_file_os + 'poems_all_v3_06.csv', low_memory=False)
    df_final = df_poems[df_poems['id'].astype(str).isin(hits)]
    df_final.loc[:, 'merge_info'] = df_final.apply(lambda row: (str(row['dynasty']) +
                                                                '·' + str(row['author'])
                                                                + '《' + str(row['title']) + '》' +
                                                                '\n诗句：' + str(row['content']) +
                                                                '。\n赏析：' + str(row['implication'])
                                                                ).replace('nan', ''), axis=1)
    poems_li = []
    ids = df_final['id'].tolist()
    for i in df_final.iloc[:]['merge_info']:
        poems_li.append(i)
    tmp_cont = ''
    for i in range(len(poems_li)):
        tmp_cont += '{poem}\n'.format(poem=poems_li[i])
    prompt_RAG = '''你是一名中国汉语言专家，在给孩子取名方面有着丰富经验。请结合用户需求、相关古诗和任务目标，取一个合适的名字，尽可能满足所有目标。最终按照输出说明给出结果。

                                任务信息：
                                - 用户需求：{user_query}
                                - 推荐知识：{poem}
                                - 任务目标：文化内涵（古诗）、父母期待、五行八字、个人特征（性别、生肖、出生年月等）、其他需求

                                 输出说明(以JSON格式)：
                                {{
                                    "名字": "...",
                                    "解释": "..."
                                }}'''.replace(' ', '')
    prompt = prompt_RAG.format(user_query=query, poem=tmp_cont)
    res, output = getResults(llm=generator, prompt=prompt, keys_=['名字', '解释'])
    if output == '':
        continue
