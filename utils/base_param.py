'''
记录文件路径
和一些基础参数
'''

import os
import re
import time
from pathlib import Path



class BaseParam:
    # GPT4o TODO: GPT模型信息
    gpt_api_key = "" # your api key
    gpt_model = "gpt-4o"
    # =================================================================================================================
    # Gemini TODO: Gemini模型信息
    gemini_api_key = '' # your api key
    gemini_model = 'gemini-1.5-flash-001'  # 2024年5月23日 轻量级
    # =================================================================================================================
    # mistralai TODO: mistral模型信息
    mistral_api_key = '' # your api key
    mistral_model = "mistral-small-latest"  # 24年7月mistral-large-latest
    # =================================================================================================================
    # kimi TODO: kimi模型信息
    kimi_api_key = ""  # your api key
    kimi_model = "moonshot-v1-8k"  # moonshot-v1-8k / moonshot-v1-32k / moonshot-v1-128k
    kimi_url = "https://api.moonshot.cn/v1"
    # =================================================================================================================
    # GLM TODO: GLM模型信息
    glm_api_key = ""  # your api key
    glm_model = "glm-4"  # glm-4 / glm-4-long / glm-4-flash
    # =================================================================================================================
    # Qwen TODO: Qwen模型信息
    qwen_api_key = '' # your api key
    qwen_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    qwen_model = 'qwen-long'  # qwen-plus, qwen-long, qwen-turbo,qwen-max
    # =================================================================================================================
    # 百川API参数 TODO: Baichuan模型信息
    baichuan_url = "https://api.baichuan-ai.com/v1/chat/completions"
    baichuan_api_key = ""  # your api key
    baichuan_model= 'Baichuan2-Turbo'
    # baichuan_model = 'Baichuan3-Turbo'
    # baichuan_model = 'Baichuan4'
    # =================================================================================================================
    # Ernie
    ernie_api_key = '' # your api key
    ernie_secret_key = '' # your secret key
    ernie_model = "ERNIE-4.0-8K"
    # =================================================================================================================
    # 讯飞API参数
    spark_appid = ""  # 填写控制台中获取的 APPID 信息
    spark_api_secret = ""  # 填写控制台中获取的 APISecret 信息
    spark_api_key = ""  # 填写控制台中获取的 APIKey 信息
    spark_domain = "general"  # Spark Lite版本
    spark_url = "wss://spark-api.xf-yun.com/v1.1/chat"  # Spark Lite环境的地址
    # =================================================================================================================

    # 当前工作目录
    data_file_os = str(Path.cwd())+'/dataset/'
    print("your current work dir:", data_file_os)
    chinese_poetry_os = data_file_os + 'chinese-poetry-master/'
    ccpc_os = data_file_os + 'CCPC/'
    famous_os = data_file_os + 'famous/'
    pdm_os = data_file_os + 'poems-db-master/'
    pc_os = data_file_os + 'parallel_corpus/'
    web_data_os = data_file_os + 'web_data/'
    poems_all_os = data_file_os + 'poems/'
    inter_data_os = poems_all_os + 'inter_data_for_v1/' # 中间步骤得到的数据
    poems_all_v1_os = poems_all_os + 'v1_base/' # 初步整合和清洗得到的数据 对应all
    poems_all_v2_os = poems_all_os + 'v2/' # 使用llm进行补充的数据 对应 all_clean
    poems_all_v3_os = poems_all_os + 'v3/' # 再次清洗的数据 对应 use
    poems_db = poems_all_os+'poems_All_use.csv'  # 最终使用的古诗文数据文件

    # 处理的一些数据文件
    # 1、Trans  ccpc_data (base 九歌)
    f_ccpc_1 = 'ccpc_test_v1.0.json'
    f_ccpc_2 = 'ccpc_train_v1.0.json'
    f_ccpc_3 = 'ccpc_valid_v1.0.json'
    f_ccpc_data_j = 'ccpc_data.json'
    f_ccpc_data_c = 'ccpc_data.csv'

    # 2、Trans  transpoem_data (base 平行语料)
    f_parallel_pairs = 'poetry_pairs.txt'
    f_parallel_wuyan = 'wuyan_par.txt'
    f_parallel_transpoem_t = 'transpoem_data.txt'
    f_parallel_transpoem_c = 'transpoem_data.csv'

    # 3、Merge&Trans famous_poems (base 名人名句)
    f_famous_t = 'final_famous.txt'
    f_famous_c = 'famous_poems.csv'

    # 4、Merge poems_TF (合并2、3)
    f_merge_tf = 'poems_TF.csv'

    # 5、Merge&Trans poems_db_fin (GitHub，古诗文)
    f_poems_db_li = ['poems1.json', 'poems2.json', 'poems3.json', 'poems4.json']
    f_merge_poems_db = 'poems_db_fin.csv'

    # 6、Trans&Merge poems_GG_only_imp (合并5和古诗句网数据中含有鉴赏的数据)
    f_gushiju = 'gushiju_poem0427.csv'  # 源数据
    f_gushiju_new = 'gushiju_poem_new0427.csv'  # 预处理格式和去重后的数据
    f_merge_gg_imp = 'poems_GG_only_imp.csv'

    # 7、Merge  poems_Base (合并1、4、5、6)
    f_poems_Base = 'poems_Base.csv'  # 合并的数据

    # 8、Trans xcz (GitHub，西窗烛)
    f_xcz = 'poem_xcz.csv'  # 从百度汉语爬取的数据
    f_xcz_new = web_data_os + f_xcz  # GitHub上由作者从西窗烛爬取得到的80w条数据

    # 9、Trans poems_base_baidu_new (poems_base之后爬取百度的解释,即运行get_baidu_data.py，得到poems_base_baidu)
    f_poems_base_baidu = 'poems_base_baidu.csv'  # 从百度汉语爬取的数据
    f_poems_base_baidu_new = 'poems_base_baidu_new.csv'

    # 10、Merge poems_BaW (合并8、9，所有base+web数据)
    f_poems_BaW = 'poems_BaW.csv'

    # 11、Merge poems_All (10增加搜韵tags）
    f_souyun_word = 'souyun_words.csv'
    f_poems_All = 'poems_All.csv'
    # =================================================================================================================
    # test
    test_os = data_file_os + 'test_res/'
    test_bm_os = test_os + 'benchmark/'
    f_test_dataset = test_bm_os + 'test_data_500.csv'
    test_bl_os = test_os + 'baselines/'
    test_magic_os = test_os + 'NAMeGEn/'
    # =================================================================================================================
    # eval
    eval_os = 'eval_res/'
    f_eval_res = eval_os + 'eval_results_{}.csv'
    f_eval_res_scores = eval_os + 'eval_results_scores.csv'
    # =================================================================================================================
    # hanzi
    hanzi_os = data_file_os + 'hanzi/'
    f_hanzi_poems = hanzi_os + 'hanzi_poems.csv'


def getLlmRes(llm, input_str, str_=''):
    res = {}
    cnt_gen = 0
    output = ''
    if str_ == '':
        while True:
            if cnt_gen > 0:
                break
            output = llm.get_answer(input_str).replace('\n', '')
            cnt_gen += 1
            try:
                res = eval(output[output.index('{'):output.rindex('}') + 1].replace('\n', '').replace(' ', ''))
                break
            except:
                continue
    else:
        while True:
            if cnt_gen > 0:
                break
            output = llm.get_answer(input_str)
            if '// ' in output:
                output = re.sub(r"//.*", "", output)
            output = output.replace('\n', '')
            cnt_gen += 1
            try:
                res = eval(output[output.index('{'):output.rindex('}') + 1].replace('\n', '').replace(' ', ''))
                s = res[str_]
                break
            except:
                continue
    return res, output

def getLlmResByMessages(llm, messages, str_=''):
    res = {}
    output = ''
    cnt_gen = 0
    if str_ == '':
        while True:
            if cnt_gen > 1:
                break
            cnt_gen += 1
            output = llm.get_answerByMessages(messages).replace('\n', '')
            try:
                res = eval(output[output.index('{'):output.rindex('}') + 1].replace('\n', '').replace(' ', ''))
                break
            except:
                continue
    else:
        while True:
            if cnt_gen > 1:
                break
            cnt_gen += 1
            output = llm.get_answerByMessages(messages).replace('\n', '')
            try:
                res = eval(output[output.index('{'):output.rindex('}') + 1].replace('\n', '').replace(' ', ''))
                s = res[str_]
                break
            except:
                continue
    # if cnt_gen > 0:
    #     # print(f'Error in getLlmResByMessages for {messages}')
    #     pass
    return res, output


# 将英文符号替换为中文符号的函数
def replace_english_symbols(text):
    return (text.replace("'", "’")
            .replace('"', '“')
            .replace(":", "：")
            .replace(",", "，")
            .replace(".", "。"))


def getResults(llm, prompt, keys_, use_his=False):
    '''
    从llm中获取名字和解释
    :param llm: llm模型
    :param prompt: 提示
    :param keys_: 填入需要解析的字典的keys
    :param wo_his: 是否使用历史记录
    '''
    cnt_gen = 0
    res = {}
    output = ''
    while True:
        if cnt_gen > 1:
            # 超过2次则跳出
            output = ''
            res = {}
            break
        try:
            cnt_gen += 1
            if not use_his:
                output = llm.get_answer(prompt)  # 不使用历史记录
            else:
                output = llm.get_answerByMessages(prompt)  # 使用历史记录
            output = output.replace('\n', '').replace(' ', '')
            # 正则表达式，使用{}起到匹配效果，提取{}内部的内容
            max_ind = len(keys_)-1
            min_ind = 0
            pattern = ""
            for i in range(len(keys_)):
                if i > min_ind and i < max_ind:
                    pattern_tmp = r"['\"‘’“”]" + keys_[i] + r"['\"‘’“”]\s*[：:]\s*['\"‘’“”]?(.*?)[\"'‘’“”]?\s*[,，]\s*"
                    pattern += pattern_tmp
                elif i == min_ind:
                    pattern_head = r"\{['\"‘’“”]" + keys_[i] + r"['\"‘’“”]\s*[：:]\s*['\"‘’“”]?(.*?)[\"'‘’“”]?\s*[,，]\s*"
                    pattern += pattern_head
                elif i == max_ind:
                    pattern_tail = r"['\"‘’“”]" + keys_[i] + r"['\"‘’“”]\s*[：:]\s*['\"‘’“”]?(.*?)['\"‘’“”]?\s*\}"
                    pattern += pattern_tail
            # 原来两个key的pattern
            # pattern = r"\{['\"‘’“”]" + keys_[0] + r"['\"‘’“”]\s*[：:]\s*['\"‘’“”]?(.*?)[\"'‘’“”]?\s*[,，]\s*['\"‘’“”]" + \
            #           keys_[1] + r"['\"‘’“”]\s*[：:]\s*['\"‘’“”]?(.*?)['\"‘’“”]?\s*\}"
            match = re.search(pattern, output)
            # 如果匹配成功，将其内容放到字典中
            if match:
                res = {}
                for i in range(len(keys_)):
                    res[keys_[i]] = replace_english_symbols(match.group(i + 1))
                break
            else:
                continue
        except:
            time.sleep(0.5)
            continue
    return res, output