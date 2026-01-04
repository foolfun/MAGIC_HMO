import json

import pandas as pd
from tools.base_param import BaseParam
params = BaseParam()

f = params.data_file_os+'Other_baselines/LLM-Discussion-main/Results/Similarities/Output/multi_agent/Similarities_multi_debate_roleplay_3_5_mistral-small-latest_古诗词文化专家-风水命理专家-现代汉语言学家_multi_agent_2024-09-04-17-45-36_500.json'
f = params.data_file_os+'Other_baselines/LLM-Discussion-main/Results/Similarities/Output/multi_agent/Similarities_multi_debate_roleplay_3_5_mistral-small-latest_古诗词文化专家-风水命理专家-现代汉语言学家_multi_agent_2024-09-04-14-01-54_500.json'
with open(f, 'r', encoding='utf-8') as file:
    data = json.load(file)
with open(f, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

with open(f, 'r') as f:
    data = json.load(f)
    print(data)


queries = []
name_1 = []
name_2 = []
final_ans = []
llm = 'mistral'
for i in data:
    q = i['question'].split('\n')[1][6:]
    if q not in queries:
        queries.append(q)
    if i['Agent'] == 'Agent 1 - 古诗词文化专家':
        anw_1 = i['answer']
        name_1 = [j['名字'] for j in anw_1]
    elif i['Agent'] == 'Agent 2 - 风水命理专家':
        anw_2 = i['answer']
        name_2 = [j['名字'] for j in anw_2]
    elif i['Agent'] == 'Agent 3 - 现代汉语言学家':
        anw_3 = i['answer']
        name_3 = [j['名字'] for j in anw_3]
        name_ = list(set(name_1) & set(name_2) & set(name_3))
        if len(name_) == 0:
            name_ = list(set(name_1) & set(name_2))
        if len(name_) == 0:
            name_ = list(set(name_1) & set(name_3))
        if len(name_) == 0:
            name_ = list(set(name_2) & set(name_3))
        # if len(name_) == 0:
        #     name_ = list(set(name_1) | set(name_2) | set(name_3))
        max_len = 0
        max_name = ''
        max_exp = ''
        for n in name_:
            a1 = [j for j in anw_1 if j['名字'] == n]
            a2 = [j for j in anw_2 if j['名字'] == n]
            a3 = [j for j in anw_3 if j['名字'] == n]
            if len(a1) == 0:
                a1 = [{'名字': n, '解释': ''}]
            if len(a2) == 0:
                a2 = [{'名字': n, '解释': ''}]
            if len(a3) == 0:
                a3 = [{'名字': n, '解释': ''}]
            a1 = a1[0]
            a2 = a2[0]
            a3 = a3[0]
            m_len = len(a1['解释'])
            exp = a1['解释']
            if m_len < len(a2['解释']):
                m_len = len(a2['解释'])
                exp = a2['解释']
            if m_len < len(a3['解释']):
                m_len = len(a3['解释'])
                exp = a3['解释']
            if max_len < m_len:
                max_len = m_len
                max_name = n
                max_exp = exp
        print(max_name, max_exp)
        final_ans.append({'query': q, 'name': max_name, 'exp': max_exp,'r_poem':None,'backbone':llm,'method':'llm_discussion','up_w':None,'output':None})

# 按照query去重
queries = []
final_ans_new = []
for i in range(len(final_ans)):
    if final_ans[i]['query'] not in queries:
        queries.append(final_ans[i]['query'])
        final_ans_new.append(final_ans[i])

df = pd.DataFrame(final_ans)
f_os = params.test_os+f'/baselines/0818/baseline_{llm}.csv'
# 将df增加到csv文件中
df.to_csv(f_os, mode='a', header=False, index=False)


# 补全up_w
f_bc = params.test_os+'benchmark/test_data_500.csv'
f_os = params.test_os+f'baselines/0818/baseline_{llm}.csv'
df_bc = pd.read_csv(f_bc)
df_gpt4o = pd.read_csv(f_os)
# df_gpt4o的up_w可以从df_bc找到与其query一样的up_w
df_bc = df_bc[['query', 'up_w']]
df_gpt4o.drop('up_w', axis=1, inplace=True)
df_gpt4o = pd.merge(df_gpt4o, df_bc, on='query', how='left')
# 重排列
df_gpt4o = df_gpt4o[['query', 'name', 'exp', 'r_poem', 'backbone', 'method', 'up_w', 'output']]
df_gpt4o.to_csv(f_os, index=False)
