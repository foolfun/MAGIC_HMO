import pandas as pd
import time
from Agents import InfoManager, Generator, Evaluator
from utils.LLMs import *
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import json
import argparse

parser = argparse.ArgumentParser(description="Run the NAMeGEn.")
parser.add_argument("-b", "--backbone", required=True, help="The backbone model to use.")
parser.add_argument("-m", "--mode", choices=["batch", "single"], default="batch",
                    help="The mode to run: single or batch.")
parser.add_argument("-n", "--number", type=int, default=-1, help="The number of queries to run.")
parser.add_argument("-ab", "--ablation", default="", help="The ablation method to choose.")
parser.add_argument("-q", "--query", type=str, default="", help="The test query for single test.")
args = parser.parse_args()

# args.backbone = 'qwen'  # 'baichuan',  'qwen', 'glm4’, 'gpt4o','mistral', 'gemini'
# args.mode = 'batch'  # 'single'
# args.number = -1
# args.ablation = "" # 'wo-R', 'wo-evalR', 'wo-Imp', 'wo-Exp', 'wo-evalGen'
# args.query = ''

params = BaseParam()
f_os_base = params.data_file_os
model = args.backbone  # 'baichuan',  'qwen', ‘glm4’, 'gpt4o','mistral', 'gemini'
num = args.number  # df_data.shape[0]
print(f'\n\n=============================Use {model}===============================\n\n')
f_ours = params.test_bl_os + f'NAMeGEn/NAMeGEn_{model}.csv'
f_ours_interRecords = params.test_bl_os + f'NAMeGEn/NAMeGEn_{model}.json'
if args.ablation != '':
    f_ours = f_ours.replace('.csv', f'_{args.ablation}.csv')
    f_ours_interRecords = f_ours_interRecords.replace('.json', f'_{args.ablation}.json')

df_exit = pd.DataFrame()
if os.path.exists(f_ours):
    df_exit = pd.read_csv(f_ours)
else:
    # 构建结果表
    df_ = pd.DataFrame(columns=['query', 'name', 'exp', 'r_poem', 'backbone', 'method', 'up_w', 'output'
        , 'f_gen_rounds', 'f_up_gen', 'f_regen_flag', 'f_rk_rounds'
        , 'f_imp_rounds', 'f_imp_s', 'f_imp_rcs', 'f_imp_t'
        , 'f_exp_rounds', 'f_exp_s', 'f_exp_rs', 'f_exp_t'
        , 'h_imp_ss', 'h_imp_rss', 'h_imp_css', 'h_imp_ts'
        , 'h_exp_ss', 'h_exp_rs', 'h_exp_ts'])

    df_.to_csv(f_ours, index=False, encoding='utf-8', header=True)
if os.path.exists(f_ours_interRecords):
    pass
else:
    with open(f_ours_interRecords, 'w+', encoding='utf-8') as file:
        pass

print('正在准备数据...')
# 读取数据
poems_file = params.poems_db
df_poems = pd.read_csv(poems_file, low_memory=False)
print('检索库中一共有{n}首古诗。'.format(n=df_poems.shape[0]))
print('数据准备完成。')


def runNAMeGEn(llm, user_input, target_kw, task_type,
               retrieval, evalR, evalGen, use_imp, use_exp):
    start_time = time.time()
    f_r_poem = ''
    tap = '*'
    num_tap = 1

    llm_am = llm
    llm_ag = llm
    llm_ae = llm

    AM = InfoManager(llm=llm_am)
    AG = Generator(llm=llm_ag)
    AE = Evaluator(llm=llm_ae)

    # step1：分析目标约束条件，意图识别
    # print('正在分析任务...')
    # 1.分析任务类型
    AM.anaTaskType(query=user_input, task_type=task_type)
    # 2.解析任务关键词(目标)
    AM.anaMOKeywords(target_kw=target_kw, evalMultiObj=AE.evalMultiObj)
    # 3.分析用户偏好
    AM.anaWeight()
    # step2: 扩展信息与检索知识
    # 2.1 扩展信息
    expand_info = AM.expandInfo()
    print(expand_info)
    # 2.2 检索相关知识
    f_r_poem = AM.getKnowledge(db=df_poems, evalRetrieval=AE.evalRetrieval,
                               flag_db=retrieval, flag_llm=False, flag_eval=evalR)
    if retrieval and (f_r_poem == ''):
        print('未检索到相关知识。')
        return
    # 2.3 细化多目标、要求
    AM.refineMO(evalMultiObj=AE.evalMultiObj)
    regen_flag = -1
    # Step3：生成与优化
    while True:
        gen_rounds = AM.gen_rounds  # 生成轮次
        if gen_rounds == 0:  # 第一次生成或退回重新生成
            print(f'第{gen_rounds + 1}次生成...')
            gen_res, output, gen_messages = AG.genResult(user_query=AM.user_query,
                                                         task_type=AM.task_type,
                                                         task_objs_cont=AM.task_objs_cont,
                                                         task_reqs_cont=AM.task_reqs_cont,
                                                         weight_dic=AM.weight_dic,
                                                         r_knowledge_cont=AM.r_knowledge_cont,
                                                         key_info_cont=AM.key_info_cont,
                                                         key_info=AM.key_info,
                                                         use_imp=use_imp,
                                                         use_exp=use_exp)  # 生成结果
            if output == '':
                print('生成结果失败...')  # 异常退出
                return
            AM.gen_messages = gen_messages
            AM.updateMemory(gen_res_li=[gen_res, output])
            print('模型输出结果：\n', gen_res)
            if not evalGen:  # 不评估生成结果, 直接结束
                final_res = AM.his_gen_res[-1]
                final_output = AM.his_outputs[-1]
                break
        elif gen_rounds > 0:
            print(f'第{gen_rounds + 1}次生成...')
            all_n = len(AM.his_gen_res)
            n = 1  # 取最近n轮的信息
            regen_flag = AM.regen_flag_li[all_n - n:][0]
            regen_res, output = AG.regenResult_new(regen_flag=regen_flag,
                                                   gen_messages=AM.gen_messages,
                                                   key_info=AM.key_info)
            if output == '':
                print('重新生成结果失败...')  # 异常退出
                return
            AM.updateMemory(gen_res_li=[regen_res, output])  # 取名结果存入memory
            print('模型输出结果：\n', regen_res)

        # step4: 评价与反馈
        print('正在评估结果...')
        feedback_dic = AE.feedback(base_info=AM.base_info, his_gen_res=AM.his_gen_res, gen_rounds=gen_rounds,
                                   use_imp=use_imp, use_exp=use_exp)  # 仅评估最新一轮生成结果

        regen_flag = feedback_dic['regen_flag']
        report = feedback_dic['feedback_report']
        AM.updateMemory(eval_res=feedback_dic)  # 取名结果存入memory
        print('模型评估的结果为：\n', regen_flag, report)

        # 依据评估结果，决定下一步操作：0-重新生成全部，1-重新生成解释，2-结束, 3-选择近期最好的结果
        if regen_flag == 0:
            print('模型输出结果不满足要求，将重新生成。')
        elif regen_flag == 1:
            print('模型输出的结果满足要求，但解释有误或不够合理，将修改解释内容。')
        elif regen_flag == 2:
            final_res = AM.his_gen_res[-1]
            final_output = AM.his_outputs[-1]
            print('模型输出结果满足要求。')
            # print('最终选择的结果为：', AM.his_gen_res[-1])
            print('总共耗时(/s)：', str(int(time.time() - start_time)))
            break
        elif regen_flag == 3:
            max_ind = feedback_dic['exp_max_ind']
            final_res = AM.his_gen_res[max_ind]
            final_output = AM.his_outputs[max_ind]
            print('模型输出结果满足要求，但不是最好的结果，将选择历史最好的结果。')
            print('最终选择的结果为：', AM.his_gen_res[max_ind])
            print('总共耗时(/s)：', str(int(time.time() - start_time)))
            break
        elif regen_flag == 4:
            print(feedback_dic['feedback_report'])  # 超过最大生成轮次，结束
            return

    # 最终结果
    final_results = {'user_preference': AM.base_info['weights'],
                     'f_res': final_res,
                     'f_output': final_output,
                     'f_gen_rounds': AM.gen_rounds,
                     'f_r_poem': f_r_poem,  # 检索到的知识
                     'f_rk_rounds': AM.rk_rounds,  # 知识检索轮次
                     'f_regen_flag': regen_flag,  # 重新生成标志
                     'f_imp_rounds': AM.eval_res_dic['imp_rounds'],  # 隐式（完整性和清晰度）的评估总次数
                     'f_imp_s': AM.eval_res_dic['imp_s'],  # 隐式得分
                     'f_imp_rcs': AM.eval_res_dic['imp_rcs'],  # 隐式细粒度得分：{'r_':[x,x,...],'rs':x,'c_':[x,x,...],'cs':x}
                     'f_imp_t': AM.eval_res_dic['imp_t'],  # 隐式阈值
                     'f_exp_rounds': AM.eval_res_dic['exp_rounds'],  # 显式评估总次数
                     'f_exp_s': AM.eval_res_dic['exp_s'],  # 显式得分
                     'f_exp_rs': AM.eval_res_dic['exp_rs'],  # 显式相关性细粒度得分：[x,x,...]
                     'f_exp_t': AM.eval_res_dic['exp_t'],  # 显式阈值
                     'exp_max_ind': AM.eval_res_dic['exp_max_ind'],  # 显式最大得分的索引
                     'h_imp_ss': AM.eval_res_dic['h_imp_ss'],  # 隐/显式的得分和阈值的所有记录 隐式得分
                     'h_imp_rss': AM.eval_res_dic['h_imp_rss'],  # 隐式完整性 [[[x,x,...],rs],[[x,x,...],rs],...]，rs为总分
                     'h_imp_css': AM.eval_res_dic['h_imp_css'],  # 隐式清晰度 [[[x,x,...],cs],[[x,x,...],cs],...]，cs为总分
                     'h_imp_ts': AM.eval_res_dic['h_imp_ts'],  # 隐式阈值
                     'h_exp_ss': AM.eval_res_dic['h_exp_ss'],  # 显式得分  [x,x,...]
                     'h_exp_rs': AM.eval_res_dic['h_exp_rs'],  # 显式相关性 [[x,x,...],[x,x,...],...]，每个方面的细粒度得分
                     'h_exp_ts': AM.eval_res_dic['h_exp_ts']}  # 显式阈值

    all_results = {
        'User_query': AM.base_info['user_query'],
        'Base_info': {
            'task_type': AM.base_info['task_type'],
            'target_kw': AM.base_info['target_kw'],
            'user_preference': AM.base_info['weights'],
            'key_info': AM.base_info['key_info'],
            'r_knowledge': AM.r_knowledge,
            'objectives': AM.base_info['task_objs'],
            'requirements': AM.base_info['task_reqs'],
        },
        'Final_results': final_results,
        'Iterative_info': {
            'task_analysis': AM.his_ana_records,
            'information_processing': AM.his_infoProcess_records,
            'MOO_process': AM.his_gen_records,
        }

    }

    return final_results, all_results


def addInterRecords(file_path, new_data):
    def ndarray_to_list(data):
        if isinstance(data, np.ndarray):
            return data.tolist()  # Convert ndarray to list
        if isinstance(data, np.int64):  # Handle numpy int64 type
            return int(data)  # Convert numpy int64 to Python int
        if isinstance(data, dict):
            # Recursively convert ndarrays and int64s in dictionaries
            return {key: ndarray_to_list(value) for key, value in data.items()}
        if isinstance(data, list):
            # Recursively convert ndarrays and int64s in lists
            return [ndarray_to_list(item) for item in data]
        return data  # Return as-is if not an ndarray or int64

    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(ndarray_to_list(new_data), ensure_ascii=False) + '\n')


def choose_backbone_llm(backbone):
    if args.ablation != '':
        method = f'{method}_{args.ablation}'
    if backbone == 'baichuan':
        llm = Baichuan()
    elif backbone == 'spark':
        llm = Spark()
    elif backbone == 'qwen':
        llm = Qwen()
    elif backbone == 'ernie':
        llm = Ernie()
    elif backbone == 'glm4' or backbone == 'glm-4-flash':
        llm = GLM()
    elif backbone == 'gpt4o' or backbone == 'gpt4o_mini':
        llm = GPT()
    elif backbone == 'mistral':
        llm = Mistral()
    elif backbone == 'gemini':
        llm = Gemini()


def choose_llm_run(backbone, query_li, up_w_li, target_kw=[], task_type='',
                   retrieval=True, evalR=True, evalGen=True, use_imp=True, use_exp=True):
    print(f'Use {backbone}')
    llm = None
    method = 'magic_moo'
    llm = choose_backbone_llm(backbone)

    for i in tqdm(range(len(query_li))):

        user_input = query_li[i]
        up_w = up_w_li[i]
        if df_exit.shape[0] > 0:
            # 如果已存在，则跳过
            df_res = df_exit[(df_exit['query'] == user_input) & (df_exit['backbone'] == backbone)]
            if df_res.shape[0] > 0:
                time.sleep(0.1)
                continue
        try:
            print(f'Query: {user_input}')
            final_results, all_results = runNAMeGEn(llm=llm,
                                                    user_input=user_input,
                                                    target_kw=target_kw,
                                                    task_type=task_type,
                                                    retrieval=retrieval,
                                                    evalR=evalR,
                                                    evalGen=evalGen,
                                                    use_imp=use_imp,
                                                    use_exp=use_exp)
        except Exception as e:
            print(user_input, e)
            continue
        res = final_results['f_res']
        print(f'Output: {res}')

        # 记录最终的生成结果、中间和最终的一些数值
        df_tmp = pd.DataFrame({'query': [user_input],
                               'name': [res['结果']],
                               'exp': [res['解释']],
                               'r_poem': [final_results['f_r_poem']],
                               'backbone': [backbone],
                               'method': [method],
                               'up_w': [up_w],
                               'output': [final_results['f_output']],
                               'f_gen_rounds': [final_results['f_gen_rounds']],  # 生成轮次
                               'f_up_gen': [final_results['user_preference']],  # 用户偏好
                               'f_regen_flag': [final_results['f_regen_flag']],  # 是否重新生成
                               'f_rk_rounds': [final_results['f_rk_rounds']],  # 检索轮次
                               'f_imp_rounds': [final_results['f_imp_rounds']],  # imp评估轮次
                               'f_imp_s': [final_results['f_imp_s']],  # 最终，imp的得分
                               'f_imp_rcs': [final_results['f_imp_rcs']],
                               # 最终，隐式细粒度得分：{'r_':[x,x,...],'rs':x,'c_':[x,x,...],'cs':x}
                               'f_imp_t': [final_results['f_imp_t']],  # 最终，imp的阈值
                               'f_exp_rounds': [final_results['f_exp_rounds']],  # exp评估轮次
                               'f_exp_s': [final_results['f_exp_s']],  # 最终，exp的得分
                               'f_exp_rs': [final_results['f_exp_rs']],  # 最终，显式相关性细粒度得分：[x,x,...]
                               'f_exp_t': [final_results['f_exp_t']],  # 最终，exp的阈值
                               'h_imp_ss': [final_results['h_imp_ss']],  # 历史，隐式得分  [x,x,...]
                               'h_imp_rss': [final_results['h_imp_rss']],
                               # 历史，隐式完整性 [[[x,x,...],rs],[[x,x,...],rs],...]，rs为总分
                               'h_imp_css': [final_results['h_imp_css']],
                               # 历史，隐式清晰度 [[[x,x,...],cs],[[x,x,...],cs],...]，cs为总分
                               'h_imp_ts': [final_results['h_imp_ts']],  # 历史，隐式阈值
                               'h_exp_ss': [final_results['h_exp_ss']],  # 历史，显式得分  [x,x,...]
                               'h_exp_rs': [final_results['h_exp_rs']],  # 历史，显式相关性 [[x,x,...],[x,x,...],...]，每个方面的细粒度得分
                               'h_exp_ts': [final_results['h_exp_ts']]})  # 历史，显式阈值 [x,x,...]
        df_tmp.to_csv(f_ours, index=False, encoding='utf-8', mode='a', header=False)
        addInterRecords(f_ours_interRecords, all_results)  # 将每轮生成结果记录到文件中
    print(f'{method} Done!')


if __name__ == '__main__':
    # 读取测试集
    df_data = pd.read_csv(params.f_test_dataset)
    if num == -1:
        num = df_data.shape[0]
    query_li = df_data['query'].values.tolist()[:num]
    r_poem_li = df_data['r_poem'].values.tolist()[:num]
    up_w_li = df_data['up_w'].values.tolist()[:num]
    task_targets = ['文化内涵（古诗）', '父母期待', '五行八字', '个人特征（性别、生肖、出生年月等）', '其他需求']
    # llm_names = ['baichuan','qwen',  'glm4', 'gpt4o', 'mistral', 'gemini']
    '''
    ablation:
    wo-retrieval: 不检索，直接生成
    wo-evalR: 检索但不评估
    wo-imp: 不使用隐式信息和评估
    wo-exp: 不使用显式信息和评估
    wo-evalGen: 不评估生成结果
    wo-ImpExp: 不使用隐式和显式信息和评估
    '''

    # 单独测试
    if args.mode == 'single':
        llm = choose_backbone_llm(backbone=model)
        user_input = args.query
        runNAMeGEn(llm=llm, user_input=user_input, retrieval=True)

    else:
        # 批量测试
        if args.ablation == '':
            # NAMeGEn
            choose_llm_run(backbone=model, query_li=query_li, up_w_li=up_w_li,
                           target_kw=task_targets, task_type='给孩子取名')

        elif args.ablation == 'wo-R':
            # wo retrieval
            print('\n\n=================wo retrieval====================\n\n')
            choose_llm_run(backbone=model, query_li=query_li, up_w_li=up_w_li,
                           target_kw=task_targets, task_type='给孩子取名',
                           retrieval=False)
        elif args.ablation == 'wo-evalR':
            # wo retrieval's evaluation
            print('\n\n=================wo evaluation retrieval====================\n\n')
            choose_llm_run(backbone=model, query_li=query_li, up_w_li=up_w_li,
                           target_kw=task_targets, task_type='给孩子取名',
                           evalR=False)
        elif args.ablation == 'wo-Imp':
            # wo implicit & imp_evaluation
            print('\n\n=================wo implicit & imp_evaluation====================\n\n')
            choose_llm_run(backbone=model, query_li=query_li, up_w_li=up_w_li,
                           target_kw=task_targets, task_type='给孩子取名',
                           use_imp=False)
        elif args.ablation == 'wo-Exp':
            # wo explicit & exp_evaluation
            print('\n\n=================wo explicit & exp_evaluation====================\n\n')
            choose_llm_run(backbone=model, query_li=query_li, up_w_li=up_w_li,
                           target_kw=task_targets, task_type='给孩子取名',
                           use_exp=False)
        elif args.ablation == 'wo-evalGen':
            # wo evaluation
            print('\n\n=================wo evaluation====================\n\n')
            choose_llm_run(backbone=model, query_li=query_li, up_w_li=up_w_li,
                           target_kw=task_targets, task_type='给孩子取名',
                           evalGen=False)
        elif args.ablation == 'wo-ImpExp':
            # wo evaluation
            print('\n\n=================wo evaluation====================\n\n')
            choose_llm_run(backbone=model, query_li=query_li, up_w_li=up_w_li,
                           target_kw=task_targets, task_type='给孩子取名',
                           evalGen=False, use_imp=False, use_exp=False)
