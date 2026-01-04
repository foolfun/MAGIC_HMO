from tqdm import tqdm
from RetrivalPoems import RetrivalPoems
from prompts.promptsGeneral import GeneralPromptDesign
from utils.base_param import getLlmRes, getLlmResByMessages, getResults
from utils.InfoProcess import GetMoreInfo
from utils.SearchHanzi import Hanzi
from copy import deepcopy
import math
import re
import numpy as np

prom_g = GeneralPromptDesign()
hz = Hanzi()
set_type = '给孩子取名'  # 给孩子取名，给中国新生儿取名
rk_limit = 5


def normalizeWeights(w_li):  # 总和为1
    w_li = list(w_li)
    total = np.sum(w_li)
    w_new = w_li / total
    w_new_rounded = [round(w, 4) for w in w_new]
    return list(w_new_rounded)


class InfoManager():
    def __init__(self, llm):
        self.llm = llm
        # 任务基本信息
        self.user_query = ''  # 固定 1 条
        self.task_type = ''  # 任务类型
        self.his_ana_records = {}  # 分析过程的历史记录
        # 扩展和细化信息
        self.key_info = {}  # 取名任务的用户信息
        self.key_info_cont = ''
        # 任务目标和要求
        self.target_kw = []  # 任务目标的关键词组合
        self.task_objs = []  # 任务目标，分点
        self.task_objs_cont = ''
        self.task_reqs = []  # 任务要求,分点
        self.task_reqs_cont = ''
        self.weights = []  # 权重
        self.weight_dic = {}  # 权重字典 {'1.xx':【分数】,'2.xx':【分数】,...}
        # 检索信息
        self.r_flag = 0  # 是否需要检索，0：不需要，1：需要
        self.r_kw = ''  # 检索描述/关键词
        self.rk_rounds = 0
        self.r_knowledge = []  # 当前检索结果
        self.r_knowledge_ids = []  # 检索结果的id,增量
        self.his_knowledge = []  # 所有历史检索结果
        self.r_knowledge_cont = ''
        self.poem_type = ''  # 古诗类型
        self.r_poem_info = []  # 古诗信息
        # self.retrival_records = []  # 检索记录
        self.his_infoProcess_records = {}  # 信息处理过程的历史记录

        # 任务基本信息(固定)
        self.base_info = {'user_query': '',  # 用户输入
                          'task_type': '',  # 任务类型
                          'key_info': {},  # 任务基本和扩展信息
                          'key_info_cont': '',  # 扩展信息
                          'target_kw': [],  # 任务目标关键词组合
                          'task_objs': [],  # 任务目标
                          'task_objs_cont': '',  # 任务目标的内容
                          'task_reqs': [],  # 任务要求
                          'task_reqs_cont': '',  # 任务要求的内容
                          'weights': [],  # 用户权重
                          'r_flag': 0,  # 是否需要检索
                          'r_info': [],  # 检索信息, [r_kw,poem_type,r_poem_info]
                          'r_knowledge': '',
                          'r_knowledge_cont': ''}  # 检索结果
        # 生成信息
        self.gen_rounds = 0  # 生成轮次
        self.his_gen_res = []  # 每个元素：{'结果':'xxx','解释':'xxx'}
        self.his_outputs = []  # 存储原结果
        self.gen_messages = []  # 生成对话 {'role':'user','content':'xxx'} or {'role':'assistant','content':'xxx'}
        # 评估信息
        self.regen_flag_li = []  # 重生成标志
        self.eval_res_dic = {
            'imp_rounds': 0,  # 隐式评估轮次
            'imp_s':0,  # 隐式评估总分
            'imp_rcs': {},  # 隐式评估细粒度得分
            'imp_t': 0.8,  # 隐式阈值
            'exp_rounds': 0,  # 显式评估轮次
            'exp_s': 0,  # 显式评估总分
            'exp_rs': [],  # 显式评估细粒度得分
            'exp_t': 0.8,  # 显式阈值
            'exp_max_ind': -1,  # 显式最大值索引
            'h_imp_ss': [],  # 隐式总分迭代记录
            'h_imp_rss': [],  # 隐式完整度得分迭代记录
            'h_imp_css': [],  # 隐式清晰度得分迭代记录
            'h_imp_ts': [],  # 隐式阈值随着迭代的动态变化记录
            'h_exp_ss': [],  # 显式总分迭代记录
            'h_exp_rs': [],  # 显式相关性细粒度得分迭代记录
            'h_exp_ts': [],  # 显式阈值随着迭代的动态变化记录
        }
        self.his_gen_records = []  # 生成和评估记录
        self.cur_gen_record = {}  # 当前生成和评估记录

    def updateMemory(self, gen_res_li=None, eval_res=None):

        def check(his_gen_res, regen_flag_li, gen_rounds):
            ls = len(his_gen_res)
            lg = len(regen_flag_li)
            if ls == lg == gen_rounds:
                print('第{n}轮信息已全更新进Memory。'.format(n=gen_rounds))

        if self.base_info['user_query'] == '':  # 更新基本信息
            self.base_info['user_query'] = self.user_query
            self.base_info['task_type'] = self.task_type
            self.base_info['key_info'] = self.key_info
            self.base_info['key_info_cont'] = self.key_info_cont
            self.base_info['target_kw'] = self.target_kw
            self.base_info['task_objs'] = self.task_objs
            self.base_info['task_objs_cont'] = self.task_objs_cont
            self.base_info['task_reqs'] = self.task_reqs
            self.base_info['task_reqs_cont'] = self.task_reqs_cont
            self.base_info['weights'] = self.weights
            self.base_info['r_info'] = [self.r_flag, self.poem_type, self.r_poem_info, self.r_kw]
            self.base_info['r_knowledge'] = self.r_knowledge
            self.base_info['r_knowledge_cont'] = self.r_knowledge_cont
            print('Memory: Base_info 已更新。')

        if gen_res_li is not None:  # 更新生成信息  gen_res_li = [gen_res, output]
            # 更新轮次
            self.gen_rounds += 1
            # 保存当前生成结果
            self.his_gen_res.append(gen_res_li[0])
            self.his_outputs.append(gen_res_li[1])
            # 当前对话记录存入cur_gen_record
            self.cur_gen_record['gen_rounds'] = self.gen_rounds
            gen_messages_temp = deepcopy(self.gen_messages)
            gen_messages_temp.append({'role': 'assistant', 'content': str(gen_res_li[1])})
            self.cur_gen_record['messages'] = gen_messages_temp  # 记录本轮对话：[最原始的输入，上一轮生成结果，上一轮的评估报告，本轮的生成结果（输出）]
            # 构建下一轮对话
            if self.gen_messages[0]['role'] == 'user':
                self.gen_messages = [self.gen_messages[0]]  # 保留最原始的输入
            elif self.gen_messages[0]['role'] == 'system':
                self.gen_messages = [self.gen_messages[1]]  # 保留最原始的输入
            self.gen_messages.append({'role': 'assistant', 'content': str(gen_res_li[1])})  # 增加本轮生成的输出
            print('Memory: Gen_info 已更新。')

        if eval_res is not None:  # 更新评估信息
            self.eval_res_dic = eval_res
            self.cur_gen_record['eval_results'] = eval_res
            cur_record_copy = deepcopy(self.cur_gen_record)
            self.his_gen_records.append(cur_record_copy)
            self.gen_messages.append({'role': 'user', 'content': eval_res['feedback_report']})  # 本轮的评估报告是下轮生成的输入
            self.regen_flag_li.append(eval_res['regen_flag'])
            print('Memory: Eval_info 已更新。')
            check(self.his_gen_res, self.regen_flag_li, self.gen_rounds)

    def anaTaskType(self, query, task_type=''):  # 1.1 分析任务类型
        self.user_query = query
        prom_ana_type = ''
        out_ = ''
        if task_type != '':
            # 若提供任务类型
            self.task_type = task_type
        else:
            # 若未提供任务类型，则自动解析
            prom_ana_type = prom_g.anaTask.format(user_query=self.user_query)
            task_type, out_ = getLlmRes(self.llm, prom_ana_type, '任务类型')
            self.task_type = task_type['任务类型']
        self.his_ana_records['type_ana'] = {'prompt': prom_ana_type, 'output': out_, 'task_type': self.task_type}

    def anaMOKeywords(self, target_kw, evalMultiObj):  # 1.2 解析任务关键词(目标)
        prom_wk = ''
        out_ = ''
        if len(target_kw) != 0:
            # 若提供关键词
            self.target_kw = target_kw
        else:
            # 若未提供关键词,则自动解析并生成关键词
            prom_wk = prom_g.genReqKey.format(task_name=self.task_type)
            target_kw, out_ = getLlmRes(self.llm, prom_wk, '任务关键词')
            self.target_kw = target_kw['任务关键词']
            while True:
                # 评估关键词设置是否合理
                report = evalMultiObj(task_name=self.task_type,
                                      user_query=self.user_query,
                                      target_kw=self.target_kw)
                if report != '':
                    # 若不合理，则重新生成
                    prom_wk = prom_g.regenReqKey.format(task_name=self.task_type,
                                                        target_kw=self.target_kw,
                                                        report=report)
                    target_kw, out_ = getLlmRes(self.llm, prom_wk, '任务关键词')
                    self.target_kw = target_kw['任务关键词']
                else:
                    break
        self.his_ana_records['kw_ana'] = {'prompt': prom_wk, 'output': out_, 'target_kw': self.target_kw}
        # print('任务关键词：', self.target_kw)

    def anaWeight(self):  # 1.3 用户偏好预测，获取权重
        prom_weight = prom_g.getMultiObjWeight.format(user_query=self.user_query,
                                                      target_kw=self.target_kw)
        self.weight_dic, out_ = getLlmRes(self.llm, prom_weight)  # 获取权重
        print('用户偏好：', self.weight_dic)
        w_li = self.weight_dic.values()
        self.weights = normalizeWeights(w_li)  # 归一化权重
        print('归一化权重：', self.weights)
        self.his_ana_records['preference_ana'] = {'prompt': prom_weight, 'output': out_, 'weights': self.weights}

    def expandInfo(self):  # 2.1 扩展信息
        if set_type in self.task_type:
            # 抽取关键信息
            prom_expand = prom_g.extraInfo_name.format(user_query=self.user_query)
            self.key_info, out_ = getLlmRes(self.llm, prom_expand, '出生时间')
            birth = self.key_info['出生时间']
            # 由出生时间来扩展信息
            ef = GetMoreInfo(birth, self.key_info['出生季节'])
            expand_info = ef.get_baby_info_new()
            self.key_info.update(expand_info)
            lunar_day = self.key_info['农历']
            shengxiao = self.key_info['生肖']
            season = self.key_info['季节']
            holiday = self.key_info['节日']
            solar_term = self.key_info['节气']
            bw = self.key_info['八字和五行']
            wx_lack = self.key_info['五行缺失']
            expand_terms = ''
            if lunar_day != '无':
                expand_terms += f"\n农历生日：{lunar_day}；"
            if shengxiao != '无':
                expand_terms += f"\n生肖：{shengxiao}；"
            if season != '无':
                expand_terms += f"\n出生季节：{season}；"
            if holiday != '无':
                expand_terms += f"\n出生节日：{holiday}；"
            if solar_term != '无':
                expand_terms += f"\n出生节气：{solar_term}；"
            if bw != '无':
                expand_terms += f"\n八字和五行：{bw}；"
                if not wx_lack:
                    expand_terms += '五行完整，无缺失元素。'
                else:
                    tmp_lack = '、'.join(wx_lack)
                    expand_terms += f"五行缺失：{tmp_lack}；"
            others_info = ''
            if self.key_info['其他信息'] != '无':
                others_info = '\n其他信息为：' + self.key_info['其他信息']
            key_info_temp = prom_g.naming_info.format(first_name=self.key_info['姓氏'],
                                                      gender=self.key_info['性别'],
                                                      meaning=self.key_info['寓意'],
                                                      birth=self.key_info['出生时间'],
                                                      expand_terms=expand_terms,
                                                      others_info=others_info)
            if key_info_temp != '':
                self.key_info_cont = '\n-扩展信息：\n{key_info}'.format(key_info=key_info_temp)
        else:
            # 其他任务类型
            prom_expand = prom_g.extraInfo_others.format(user_query=self.user_query,
                                                         task_name=self.task_type)
            key_, out_ = getLlmRes(self.llm, prom_expand, '关键信息')
            self.key_info = key_['关键信息']
            if self.key_info != '':
                self.key_info_cont = '\n-扩展信息：\n{key_info}'.format(key_info=self.key_info)
        self.his_infoProcess_records['expand_info'] = {'prompt': prom_expand, 'output': out_, 'key_info': self.key_info}
        return self.key_info_cont

    def refineMO(self, evalMultiObj):  # 2.3 细化目标与要求
        refine_recodes = []
        prom_req = prom_g.genMultiObj.format(user_query=self.user_query,
                                             key_info_cont=self.key_info_cont,
                                             r_knowledge_cont=self.r_knowledge_cont,
                                             target_kw=self.target_kw,
                                             objs_num=len(self.target_kw))
        temp_messages = [{'role': 'user', 'content': prom_req}]
        res, out_ = getLlmRes(self.llm, prom_req, '任务目标')
        self.task_objs = res['任务目标']
        self.task_reqs = res['任务要求']
        self.task_reqs_cont = "请返回合适的结果，并给出相应的解释，其中解释应包含以下内容。\n" + '\n'.join(self.task_reqs)
        # 生成任务目标内容
        self.task_objs_cont = '\n'.join(self.task_objs)
        refine_recodes.append({
            'prompt': prom_req, 'output': out_, 'task_objs': self.task_objs, 'task_reqs': self.task_reqs
        })
        while True:
            # 评估目标设置是否合理
            temp_messages = temp_messages[:1] + [{'role': 'assistant', 'content': str(res)}]
            if (len(self.task_objs) != len(self.target_kw)) or (len(self.task_reqs) != len(self.target_kw)):
                report = '任务目标和任务要求的数量不符，请返回合适的结果。'
            else:
                report = evalMultiObj(task_name=self.task_type,
                                      user_query=self.user_query,
                                      target_kw=self.target_kw,
                                      key_info_cont=self.key_info_cont,
                                      r_knowledge_cont=self.r_knowledge_cont,
                                      task_objs_cont=self.task_objs_cont)
            temp_messages.append({'role': 'user', 'content': report})
            if report != '':
                res, out_ = getLlmResByMessages(self.llm, temp_messages, '任务目标')
                self.task_objs = res['任务目标']
                self.task_reqs = res['任务要求']
                self.task_reqs_cont = "请返回合适的结果，并给出相应的解释，其中解释应包含以下内容。\n" + '\n'.join(
                    self.task_reqs)
                self.task_objs_cont = '\n'.join(self.task_objs)
                refine_recodes.append({'messages': temp_messages, 'output': out_, 'task_objs': self.task_objs,
                                       'task_reqs': self.task_reqs})
            else:
                # 若合理，则退出评估
                print('任务目标合理。')
                break
        print('任务目标：\n', self.task_objs_cont, '\n任务要求：\n', self.task_reqs_cont)
        self.his_infoProcess_records['refine_MO'] = refine_recodes

    def genRetrivalDescription(self, flag_llm):  # 生成检索描述
        if flag_llm:
            # 使用llm进行检索时，生成检索关键词
            get_rekey = prom_g.getRetrievalKey0.format(user_query=self.user_query, key_info_cont=self.key_info_cont)
            self.r_kw, out_ = getLlmRes(self.llm, get_rekey, '检索关键词')
            self.r_flag = self.r_kw['是否检索']
            self.r_kw = self.r_kw['检索关键词']
            print('是否检索：', self.r_flag)
            print('检索关键词：', self.r_kw)
        else:
            # 使用检索库进行检索时，生成检索描述
            self.r_flag = 1  # 默认检索
            # 4,获取检索描述 {古诗类型，古诗信息，检索描述}
            if '生肖' in self.user_query:
                self.user_query = self.user_query + '孩子的生肖为' + self.key_info['生肖'] + '。'

            get_rekey = prom_g.getRetrievalKeyName.format(user_query=self.user_query)
            self.r_kw, out_ = getResults(llm=self.llm, prompt=get_rekey, keys_=['古诗类型', '古诗信息', '检索描述'])
            self.poem_type = self.r_kw['古诗类型'] if self.r_kw['古诗类型'] != '无' else ''
            self.r_poem_info = eval(
                str(self.r_kw['古诗信息']).replace('‘', '"').replace('’', '"').replace('，', ',').replace('“',
                                                                                                         '\'').replace(
                    '”', '\''))
            self.r_kw = self.r_kw['检索描述']
            print('古诗类型：', self.poem_type)
            print('古诗信息：', self.r_poem_info)
            print('检索描述：', self.r_kw)
        return get_rekey, out_

    def getKnowledge(self, db, evalRetrieval, flag_db=True, flag_llm=False, flag_eval=True):  # 2.2 检索知识
        '''
        flag_db: 是否需要用知识库检索
        flag_llm: 是否使用LLMs进行检索
        flag_eval: 是否评估检索结果
        '''
        if not flag_db:  # 不需要检索
            return ''
        prompt_, out_ = self.genRetrivalDescription(flag_llm)  # 生成检索描述
        retrival_records = [{
            'prompt': prompt_, 'output': out_,
            'r_info': [self.r_flag, self.poem_type, self.r_poem_info, self.r_kw, '']
        }]
        self.rk_rounds = 1
        while True:
            if flag_llm:
                # 激发LLMs的自身的潜在知识
                print(f'任务为：{self.task_type}。正在激发LLMs的潜在能力以获取知识...')
                self.retrieveFromLLMs(self.rk_rounds)
            else:
                # 使用检索库进行检索
                print(f'任务为：{self.task_type}。正在检索知识库以获取知识...')
                if flag_eval:
                    self.getPoems(db=db, num=5, rk_rounds=self.rk_rounds)
                else:
                    self.getPoems(db=db, num=1, rk_rounds=self.rk_rounds)
            retrival_records[-1]['r_knowledge'] = self.r_knowledge  # 更新检索记录
            if not flag_eval: # 若不需要评估，则直接返回
                break
            # 评估检索结果
            reRetrival_flag, reRetrival_key, r_knowledge_num, prompt_, out_ = evalRetrieval(task_name=self.task_type,
                                                                                            user_query=self.user_query,
                                                                                            r_kw=self.r_kw,
                                                                                            r_knowledge_cont=self.r_knowledge_cont,
                                                                                            rk_rounds=self.rk_rounds)
            self.r_kw = reRetrival_key  # 更新检索描述
            retrival_records.append({
                'prompt': prompt_, 'output': out_,
                'r_info': [reRetrival_flag, self.poem_type, self.r_poem_info, self.r_kw, r_knowledge_num]
            })
            if reRetrival_flag == 1:
                # 重新检索
                self.rk_rounds += 1  # 轮次加1
                if self.rk_rounds <= rk_limit:
                    print(f'检索不合理，更新的检索描述为：{self.r_kw} 重新检索。第{self.rk_rounds}轮...')
                else:
                    print('超过5轮，尝试从历史知识中选取最佳知识。')
            else:
                # 不需要重新检索，则构建检索结果
                try:
                    if '#' in r_knowledge_num:
                        r_num = int(r_knowledge_num.replace('#', ''))
                    else:
                        r_num = int(r_knowledge_num)
                    self.r_knowledge = self.r_knowledge[r_num - 1]
                    self.r_knowledge_cont = f'\n-推荐知识：\n{self.r_knowledge}'
                    break
                except:
                    print('选取检索知识失败，重新选取...')
                    if self.rk_rounds > rk_limit + 1:
                        print('已经尝试多次检索知识，但未能选取到合适的知识。')
                        self.r_knowledge_cont = ''
                        break
                    self.rk_rounds += 1
                    continue

        print('检索完成。最终选择的检索知识为：', self.r_knowledge_cont)
        self.his_infoProcess_records['retrival_knowledge'] = retrival_records
        return self.r_knowledge

    def getPoems(self, db, num, rk_rounds):  # 2.2 检索知识-古诗
        '''
        db: 数据库
        num: 检索数量
        rk_rounds: 当前检索轮次
        '''
        if rk_rounds <= rk_limit:
            # 小于等于5轮时，使用检索库，检索古诗
            rp = RetrivalPoems(self.key_info, self.r_kw)  # 检索对象
            tmp_ids = []
            if self.r_poem_info:
                # 有古诗信息，半精确查找
                poems_li = rp.search_poems(df=db, related_info=self.r_poem_info)
            else:
                # 无古诗信息，嵌入式查找
                poems_li, tmp_ids = rp.get_poems(df_poems=db, poem_type=self.poem_type,
                                                 ids=self.r_knowledge_ids, num=num)
            self.r_knowledge = poems_li
            self.his_knowledge.extend(poems_li)
            self.r_knowledge_ids.extend(tmp_ids)
        else:
            # 大于5轮时，返回历史知识
            self.r_knowledge = self.his_knowledge
        # 组织检索结果
        tmp_cont = ''
        for i in range(len(self.r_knowledge)):
            tmp_cont += '#{num}：{poem}\n'.format(num=i + 1, poem=self.r_knowledge[i])
        if tmp_cont != '':
            self.r_knowledge_cont = '\n-推荐知识：\n{r_knowledge}'.format(r_knowledge=tmp_cont)

    def retrieveFromLLMs(self, rk_rounds):  # 2.2 检索知识-LLMs
        if rk_rounds < rk_limit:
            prom_re = prom_g.getRetrievalInfo.format(key_words=self.r_kw)
            r_knowledge, _ = getLlmRes(self.llm, prom_re, '检索结果')
            self.r_knowledge = r_knowledge['检索结果']
            self.his_knowledge.extend(r_knowledge['检索结果'])
        else:
            self.r_knowledge = self.his_knowledge

        tmp_cont = ''
        for i in range(len(self.r_knowledge)):
            tmp_cont = '#{}：{}\n'.format(i + 1, self.r_knowledge[i])
        if tmp_cont != '':
            self.r_knowledge_cont = '\n-推荐知识：\n{r_knowledge}'.format(r_knowledge=tmp_cont)


class Generator():
    def __init__(self, llm):
        self.llm = llm

    # step3: 生成结果和解释，输入：query 和 生成目标
    def getRes(self, prompt, use_his=False):  # 生成结果和解释
        final_result = {}
        cnt_gen = 0
        output = ''
        while True:
            if cnt_gen > 2:  # 超过三次则跳出
                output = ''
                final_result = {}
                break
            try:
                cnt_gen += 1
                if use_his == False:
                    output = self.llm.get_answer(prompt)
                else:
                    output = self.llm.get_answerByMessages(prompt)
                output = output.replace('\n', '').replace(' ', '')
                pattern = r'"结果":"(.*?)"'
                match = re.search(pattern, output)
                if match is None:
                    pattern2 = r'"结果":“(.*?)”'
                    match = re.search(pattern2, output)
                result_value = match.group(1)
                pattern = r'"(\d+)\.(.*?)":"(.*?)"'
                matches = re.findall(pattern, output, re.DOTALL)
                if len(matches) == 0:
                    pattern2 = r"'(\d+)\.(.*?)':'(.*?)'"
                    matches = re.findall(pattern2, output, re.DOTALL)
                explanation = {f"{match[0]}.{match[1]}": match[2].replace("'", "‘").replace('"', '“') for match in
                               matches}
                final_result = {"结果": result_value, "解释": explanation}
                break
            except:
                continue
        return final_result, output

    def genResult(self, user_query, task_type,
                  task_objs_cont, task_reqs_cont, weight_dic,
                  r_knowledge_cont, key_info_cont, key_info, use_imp, use_exp):
        scores = {4: '4分，非常关注', 3: '3分，较关注', 2: '2分，一般关注', 1: '1分，较不关注', 0: '0分，不关注'}
        weight_dic_ = {k: scores[v] for k, v in weight_dic.items()}
        if use_imp:
            task_reqs_cont = '\n-任务要求：\n' + task_reqs_cont
        else:
            task_reqs_cont = ''
        if use_exp:
            task_objs_cont = '\n-任务目标：\n' + task_objs_cont
        else:
            task_objs_cont = ''
        prompt_gen = prom_g.genCont.format(task_name=task_type,
                                           user_query=user_query,
                                           key_info_cont=key_info_cont,
                                           r_knowledge_cont=r_knowledge_cont,
                                           task_objs_cont=task_objs_cont,
                                           task_reqs_cont=task_reqs_cont,
                                           weight_dic_=weight_dic_)
        gen_messages = [{'role': 'user', 'content': prompt_gen}]
        final_result, output = self.getRes(prompt=prompt_gen)
        if final_result['结果'].startswith(key_info['姓氏']):
            pass
        else:
            final_result['结果'] = key_info['姓氏'] + final_result['结果']
        print(output)
        return final_result, output, gen_messages

    def regenResult_new(self, regen_flag, gen_messages, key_info):  # 重新生成,依据评价重新生成
        re_result = {}
        output = ''
        if regen_flag == 0:  # 重新生成结果
            re_result, output = self.getRes(prompt=gen_messages, use_his=True)
        elif regen_flag == 1:  # 重新生成解释
            re_result, output = getLlmResByMessages(self.llm, gen_messages)
        if re_result['结果'].startswith(key_info['姓氏']):
            pass
        else:
            re_result['结果'] = key_info['姓氏'] + re_result['结果']
        return re_result, output


class Evaluator():
    def __init__(self, llm):
        self.llm = llm
        # 隐式当前
        self.imp_s = 0  # 当前：隐式评估总分
        self.imp_rcs = {}  # 当前：{'r_':[x,x,...],'rs':x,'c_':[x,x,...],'cs':x}
        # 隐式迭代记录
        self.h_imp_ss = []  # 隐式总分迭代记录 [x,x,...]
        self.h_imp_rss = []  # 隐式完整度得分迭代记录 [[[x,x,...],rs],[[x,x,...],rs],...]，rs为总分
        self.h_imp_css = []  # 隐式清晰度得分迭代记录 [[[x,x,...],cs],[[x,x,...],cs],...]，cs为总分
        self.h_imp_ts = []  # 隐式阈值随着迭代的动态变化记
        # 显式当前
        self.exp_s = 0  # 当前：显式评估总分
        self.exp_rs = []  # 当前：显式相关性细粒度得分 [x,x,...]
        # 显式迭代记录
        self.h_exp_ss = []  # 显式总分迭代记录 [x,x,...]
        self.h_exp_rs = []  # 显式相关性细粒度得分迭代记录 [[x,x,...],[x,x,...],...]
        self.h_exp_ts = []  # 显式阈值随着迭代的动态变化记录
        # 评估报告
        self.reports_dic = {'imp_report': '', 'exp_report': ''}
        self.exp_max_ind = -1  # 最大值索引
        # 阈值设计
        self.max_round = 10  # 最大轮次
        self.warmup_round = 2  # 预热轮次
        self.lr = 0.8  # 初始阈值
        self.alpha = 0.8  # 阈值衰减系数
        self.threshold = self.initTValue()  # 阈值列表
        self.imp_rounds = 0
        self.exp_rounds = 0
        self.imp_t = self.threshold[self.imp_rounds]  # 初始隐式阈值
        self.exp_t = self.threshold[self.exp_rounds]  # 初始显式阈值

    def resetCurValue(self):
        self.imp_s = 0
        self.imp_rcs = {}
        self.exp_s = 0
        self.exp_rs = []
        self.reports_dic = {'imp_report': '', 'exp_report': ''}

    def initTValue(self):
        threshold = []
        for cur_step in range(self.max_round):
            if cur_step < self.warmup_round:
                threshold.append(self.lr)
            else:
                threshold.append(self.lr / (self.alpha * math.log(cur_step + self.warmup_round)))
        return threshold

    def updateThreshold(self):
        self.imp_t = self.threshold[self.imp_rounds]
        self.exp_t = self.threshold[self.exp_rounds]

    def integrateEval(self, report, regen_flag):  # 整合评估结果
        '''
        整合评估结果
        :param report: 反馈给生成代理的内容
        :param regen_flag: 生成标志,0：重新生成结果,1：重新生成解释,2：通过
                            3：超过一定次数选近期最好的显式结果，4：超过最大轮次且历史没有显式结果，异常退出
        '''

        if regen_flag == 0:
            new_report = "请依据以下反馈进行修改，注意保持解释的通顺与合理，并按照原需求的JSON格式返回。\n" + report
            # new_report = "请依据以下反馈进行修改，请深吸一口气，一步一步考虑这些建议，若建议与任务目标或要求冲突可不接受，并按照原需求的JSON格式整理结果。\n" + report
            # new_report = "请依据以下反馈进行修改，注意保持解释的通顺与合理。*注意：若需要重写结果，请结合所给建议和背景信息，参考示例写出分析过程，并按照原需求的JSON格式整理结果。\n" + report
            # new_report = "请依据任务信息（包括，用户需求，推荐知识，任务目标，任务要求等），判断下列哪些建议是可接受的（如果与任务信息冲突则视为不可接受，如，取名任务中，任务目标中包含需要考虑从推荐知识中凝练出名字，而所给建议超出其推荐知识的范围则视为不可接受的建议），并利用这些可行的建议来调整原结果或解释。请深吸一口气，一步一步考虑这些建议，并按照原需求的JSON格式整理结果。\n" + report
        elif regen_flag == 1:
            new_report = '请保留"结果"，并依据以下反馈修改"解释"。注意保持解释的通顺与合理，并按照原需求的JSON格式返回。\n' + report
        else:
            new_report = report
        feedback_dic = {
            'regen_flag': regen_flag,
            'feedback_report': new_report,  # 评估报告
            'reports_dic': self.reports_dic,
            'imp_rounds': self.imp_rounds,
            'imp_s': self.imp_s,  # 隐式
            'imp_rcs': self.imp_rcs,
            'imp_t': self.imp_t,
            'h_imp_ss': self.h_imp_ss,
            'h_imp_rss': self.h_imp_rss,
            'h_imp_css': self.h_imp_css,
            'h_imp_ts': self.h_imp_ts,
            'exp_rounds': self.exp_rounds,
            'exp_s': self.exp_s,  # 显式
            'exp_rs': self.exp_rs,
            'exp_t': self.exp_t,
            'exp_max_ind': self.exp_max_ind,
            'h_exp_ss': self.h_exp_ss,
            'h_exp_rs': self.h_exp_rs,
            'h_exp_ts': self.h_exp_ts
        }
        return feedback_dic

    def evalMultiObj(self, task_name, user_query, target_kw,
                     key_info_cont='', r_knowledge_cont='',
                     task_objs_cont=''):  # 评估目标解析和目标填充是否合理
        if task_objs_cont == '':  # 评估目标解析是否合理
            report = ''
            prom_eval_mok = prom_g.evalMutilObjKw.format(task_name=task_name,
                                                         user_query=user_query,
                                                         target_kw=target_kw)
            # 1.1 评估：目标设置是否合理
            print('正在评估任务关键词是否合理...')
            mk_c, _ = getLlmRes(self.llm, prom_eval_mok, '是否合理')
            flag_c = mk_c['是否合理']
            if flag_c == 0 or flag_c == '0':  # 0:目标设置不合理
                report += '关键词设置不合理。修改建议如下，{}\n'.format(mk_c['修改建议'])
        else:  # 评估：目标内容细化是否合理
            report = ''
            prom_eval_mo = prom_g.evalMutilObj.format(task_name=task_name,
                                                      target_kw=target_kw,
                                                      key_info_cont=key_info_cont,
                                                      r_knowledge_cont=r_knowledge_cont,
                                                      objs_num=len(target_kw),
                                                      user_query=user_query,
                                                      task_objs_cont=task_objs_cont)
            # 1.2 评估目标设置是否合理
            print('正在评估任务目标拆解是否合理...')
            mo_c, _ = getLlmRes(self.llm, prom_eval_mo, '合理性')
            flag_c = mo_c['合理性']
            if flag_c == 0 or flag_c == '0':  # 0:目标设置不合理
                report += '任务目标拆解不合理。修改建议如下，{}\n'.format(mo_c['修改建议'])
                print('任务目标拆解不合理。')
        return report

    def evalRetrieval(self, task_name, user_query, r_kw, r_knowledge_cont, rk_rounds):
        reRetri_flag = 0  # 是否重新检索,1:重新检索,0:不重新检索
        re_retri_key = ''  # 新的检索描述
        if rk_rounds < rk_limit:
            prom_eval_retrieval = prom_g.evalRetrieval.format(task_name=task_name,
                                                              user_query=user_query,
                                                              r_kw=r_kw,
                                                              r_knowledge_cont=r_knowledge_cont)
            print('正在评估检索是否合理......')
            r_retri, out_ = getLlmRes(self.llm, prom_eval_retrieval, '是否重新检索')
            print(r_knowledge_cont)
            print(out_)
            f_retri = r_retri['是否重新检索']
            if f_retri == 1 or f_retri == '1':
                re_retri_key = r_retri['检索描述']
                reRetri_flag = 1
                # print('检索不合理，需要重新检索。')
        else:
            prom_eval_retrieval = prom_g.evalRetrieval_last.format(user_query=user_query,
                                                                   r_knowledge_cont=r_knowledge_cont)
            r_retri, out_ = getLlmRes(self.llm, prom_eval_retrieval, '最相关的知识')

        return reRetri_flag, re_retri_key, r_retri['最相关的知识'], prom_eval_retrieval, out_

    def evalACC(self, task_name, user_query, key_info, key_info_cont, r_knowledge_cont, exp_li, gen_cont):
        report = ''
        exp_tmp = deepcopy(exp_li)
        if task_name in set_type:
            print('字数...')
            first_name = key_info['姓氏']
            cnt_name = len(gen_cont[len(first_name):])
            if cnt_name > 2:
                # report += f'‘{gen_cont[len(first_name):]}’的字数为{cnt_name}。按照一般中国起名习惯，除姓氏‘{first_name}’外，剩下的名字字数应为1-2个字。请修正。'
                report += f'‘{gen_cont[len(first_name):]}’的字数为{cnt_name}。按照一般中国起名习惯，除姓氏‘{first_name}’外，剩下名字的字数应为1-2个字，少数情况下字数为3。请修正。'

            print('五行...')
            prom_extra_wx = prom_g.extraWuxing.format(exp=exp_tmp[2])
            exp_tmp.pop(2)
            r_wx, _ = getLlmRes(self.llm, prom_extra_wx)
            h_li = r_wx.keys()
            h_wx = {h: hz.getWuxingByHanzi(h) for h in h_li}
            r_wx_acc = ''  # 空表示正确，非空表示错误
            for h in h_li:
                if r_wx[h] != h_wx[h]:
                    r_wx_acc += f'“{h}五行属性为{r_wx[h]}”解释错误，正确结论为“{h}五行属性为{h_wx[h]}”，请修正。'
            if r_wx_acc != '':
                report += '五行属性解释有误，' + r_wx_acc

        # print('内容与逻辑...')
        # prom = prom_g.evalAcc.format(user_query=user_query,
        #                              key_info_cont=key_info_cont,
        #                              r_knowledge_cont=r_knowledge_cont,
        #                              gen_cont=gen_cont,
        #                              exp='\n'.join(exp_tmp))
        # res, _ = getLlmRes(self.llm, prom, '结论与建议')
        # if res['正确性'] == 0:
        #     report += f"{res['结论与建议']}"
        #     report = '解释中出现了一些错误，请修正。具体修改建议如下，' + report

        return report

    def evalImpMOs(self, task_name, user_query, target_kw, task_reqs_li,
                   key_info, key_info_cont, r_knowledge_cont, exp_li, gen_cont):
        '''
        评估隐式多目标：与输出文本属性相关的多目标优化评估，包括是否准确（事实性）、是否完成任务设定的要求（完整性）、是否清晰（清晰度）
        :param task_name: 任务类型
        :param user_query: 用户输入
        :param task_reqs_li: 任务要求列表
        :param key_info_cont: 关键信息内容
        :param r_knowledge_cont: 检索结果内容
        :param exp_li: 解释列表
        :param gen_cont: 生成结果
        :return: regen_flag, report
        '''

        imp_report = ''
        # 1.必须通过基于规则的正确性评估，才能进行后续评估
        print('正在评估正确性...')
        report_acc = self.evalACC(task_name, user_query, key_info, key_info_cont, r_knowledge_cont, exp_li, gen_cont)
        self.reports_dic['imp_acc_report'] = report_acc
        if report_acc != '':
            imp_report = report_acc
            self.h_imp_ss.append(None)
            self.h_imp_rss.append(None)
            self.h_imp_css.append(None)
            self.h_imp_ts.append(self.imp_t)
            return 0, imp_report

        # 2.评估是否完成任务要求和是否清晰
        print('正在评估任务要求完成度和清晰度...')
        res_req_li = []
        res_cla_li = []
        for i in tqdm(range(len(task_reqs_li))):
            try:
                prom = prom_g.evalCla.format(task_name=task_name,
                                             task_req_i=task_reqs_li[i][2:],
                                             exp_i=exp_li[i][2:])
                res, _ = getLlmRes(self.llm, prom, '完成度')
                res_req = res['完成度']
                res_cla = res['清晰度']
                report_ = res['结论与建议']
            except:
                res_req = 0
                res_cla = 0
                report_ = '缺少该方面的解释。'
            res_req_li.append(res_req)
            res_cla_li.append(res_cla)
            if res_req < 2 or res_cla < 2:
                imp_report += f"-在{target_kw[i]}方面：{report_}\n"

        # 3.隐式多目标优化转换
        req_s = np.sum(res_req_li) / (2 * len(res_req_li))  # 完成度,加和平均得分（归一化0-1）
        cla_s = np.sum(res_cla_li) / (2 * len(res_cla_li))  # 清晰度,加和平均得分（归一化0-1）
        impR = np.array([req_s, cla_s])  # 完成度、清晰度 - 隐式目标
        impW = np.array([0.5, 0.5])
        f_moo_imp = np.round(np.dot(impR, impW), 4)  # 多目标优化的评价函数
        # 记录数值
        self.imp_s = f_moo_imp
        self.imp_rcs = {'r_': res_req_li, 'rs': req_s, 'c_': res_cla_li, 'cs': cla_s}
        self.h_imp_ss.append(self.imp_s)
        self.h_imp_rss.append([res_req_li, req_s])
        self.h_imp_css.append([res_cla_li, cla_s])
        self.h_imp_ts.append(self.imp_t)
        print('隐式多目标结果：', f_moo_imp)
        print('历史评估结果：', self.h_imp_ss)
        self.imp_rounds += 1

        if f_moo_imp < self.imp_t:
            regen_flag = 0
            imp_report = '解释在某些方面未满足任务要求或不够清晰。具体内容如下，\n' + imp_report
        else:
            regen_flag = 2

        return regen_flag, imp_report

    def evalExpMOs(self, task_name, user_query,
                   target_kw, weights, task_objs_li, task_objs_cont,
                   exp_li, exp_li_t, gen_cont, r_knowledge_cont):
        '''
        评估显式多目标：与任务类型相关的多目标优化评估
        :param task_name: 任务类型
        :param user_query: 用户输入
        :param target_kw: 目标关键词
        :param task_objs_li: 任务目标
        :param task_objs_cont: 任务目标内容
        :param exp_li_t: 解释列表（仅内容，无编号）
        :param exp_li: 解释列表（全）
        :param weights: 权重
        :param gen_cont: 生成结果（不包含解释）
        :return: regen_flag, report
        '''

        # def template_rel(rd):
        #     s_ = ''
        #     if rd['相关性'] == 0 or rd['相关性'] == '0':
        #         s_ = '完全不相关。'
        #     elif rd['相关性'] == 1 or rd['相关性'] == '1':
        #         s_ = '部分相关。'
        #     s_ += '{}\n'.format(rd['结论与建议'])
        #     return s_

        rel_li = []  # 相关性
        rel_report = ''
        for i in tqdm(range(len(target_kw))):
            if weights[i] > 0:
                try:
                    promExp_rel = prom_g.evalExplainRel.format(task_name=task_name,
                                                               gen_cont=gen_cont,
                                                               r_knowledge_cont=r_knowledge_cont,
                                                               target_kw_i=target_kw[i],
                                                               task_objs_i=task_objs_li[i].split('：')[1],
                                                               exp_i=exp_li_t[i][2:])
                    r_exp_rel, _ = getLlmRes(self.llm, promExp_rel, '相关性')  # {'相关度':'xxx','评估结论':'xxx'}
                    f_exp_rel = r_exp_rel['相关性']
                    rel_li.append(f_exp_rel)
                    if f_exp_rel < 2:
                        rel_report += f'-在{target_kw[i]}方面：' + r_exp_rel['结论与建议']
                except:
                    rel_li.append(0)
                    rel_report += f'-{target_kw[i]}：缺少该方面的解释。\n'
            else:
                rel_li.append(0)

        # 显示多目标优化转换
        expR = np.array(rel_li)  # 相关度 - 显式目标
        n_expR = (expR - 0) / (2 - 0)
        expW = np.array(weights)  # 固定的用户偏好，作为权重
        f_moo_exp = np.round(np.dot(expW, n_expR), 4)  # 多目标优化的评价函数
        # 记录数值
        self.exp_s = f_moo_exp
        self.exp_rs = expR
        print('相关性评估结果：', f_moo_exp)
        print('历史评估结果：', self.h_exp_ss)
        try:
            h_exp_ss_clean = [-np.inf if x is None else x for x in self.h_exp_ss]
            self.exp_max_ind = np.argmax(h_exp_ss_clean)  # 最大值索引
            print('历史最大评估结果：', self.h_exp_ss[self.exp_max_ind])
        except:
            pass
        # 将当前数值存入历史记录中
        self.h_exp_rs.append(self.exp_rs)
        self.h_exp_ss.append(self.exp_s)
        self.h_exp_ts.append(self.exp_t)
        self.exp_rounds += 1 # 显式评估轮次加1
        report = ''
        self.reports_dic['exp_rel_report'] = rel_report
        if f_moo_exp < self.exp_t:
            # 多目标优化评估不通过
            if (self.exp_max_ind != -1) and (self.h_exp_ss[self.exp_max_ind] is not None) and (
                    self.h_exp_ss[self.exp_max_ind] > self.exp_t):  # 历史最大值大于当前阈值
                report = f'显式评估结果不通过，但存在历史结果大于{self.exp_t}，请参考历史结果。'
                regen_flag = 3
            else:
                promExp = prom_g.sumExp.format(task_name=task_name,
                                               user_query=user_query,
                                               task_objs_cont=task_objs_cont,
                                               gen_cont=gen_cont,
                                               exp='\n'.join(exp_li),
                                               report=rel_report)
                res, _ = getLlmRes(self.llm, promExp, '重写结果')
                if res['重写结果'] == 1 or res['重写结果'] == '1':
                    regen_flag = 0
                    report = res['结论与建议']  # 如果需要重写，返回重写的评估结论
                else:
                    regen_flag = 1
        else:
            regen_flag = 2  # 满足多目标要求
        return regen_flag, report

    def feedback(self, base_info, his_gen_res, gen_rounds, use_imp, use_exp):  # step4: 评价&优化结果
        # 新一轮评估的初始化
        self.resetCurValue()  # 重置当前显式和隐式评估值、以及评估报告
        self.updateThreshold()  # 更新阈值
        # 获取基本信息
        task_name = base_info['task_type']
        user_query = base_info['user_query']
        key_info = base_info['key_info']
        key_info_cont = base_info['key_info_cont']
        target_kw = base_info['target_kw']
        task_objs_li = base_info['task_objs']
        task_objs_cont = base_info['task_objs_cont']
        task_reqs_li = base_info['task_reqs']
        task_reqs_cont = base_info['task_reqs_cont']
        weights = base_info['weights']
        r_knowledge_cont = base_info['r_knowledge_cont']
        gen_cont = his_gen_res[-1]['结果']
        exp = his_gen_res[-1]['解释']
        exp_li = [str(k) + ":" + str(v) for k, v in exp.items()]
        exp_li_t = [str(v) for k, v in exp.items()]

        if gen_rounds <= 10:
            # step1: 评估隐式多目标
            # print('整体评估开始...')
            # prompt = prom_g.evalRough.format(task_name=task_name,
            #                            user_query=user_query,
            #                            r_knowledge_cont=r_knowledge_cont,
            #                            task_objs_cont=task_objs_cont,
            #                            task_reqs_cont=task_reqs_cont,
            #                            gen_cont_all=str(his_gen_res[-1]))
            # res,out_ = getLlmRes(self.llm, prompt, '表现')
            # if res['表现'] < 3:
            #     regen_flag = 0
            #     self.h_exp_rs.append(None)
            #     self.h_exp_ss.append(None)
            #     self.h_exp_ts.append(self.exp_t)
            #     print('整体评估不通过，需要重新生成。')
            #     report = res['结论与建议']
            #     self.reports_dic['rough_report'] = report
            #     return self.integrateEval(report, regen_flag)
            if use_imp:
                regen_flag, imp_report = self.evalImpMOs(task_name, user_query, target_kw, task_reqs_li,
                                                         key_info, key_info_cont, r_knowledge_cont, exp_li,
                                                         gen_cont)
                self.reports_dic['imp_report'] = imp_report
            else:
                imp_report = ''
                regen_flag = 2
            report = imp_report
            if regen_flag == 0:
                regen_flag = 0
                self.h_exp_rs.append(None)
                self.h_exp_ss.append(None)
                self.h_exp_ts.append(self.exp_t)
                print('隐式多目标评估不通过，需要重新生成。')
            elif regen_flag == 1:
                self.h_exp_rs.append(None)
                self.h_exp_ss.append(None)
                self.h_exp_ts.append(self.exp_t)
                print('隐式多目标评估不通过，需要修改解释。')
            elif regen_flag == 2 and use_exp:
                # step2：评估显式多目标
                print('隐式多目标评估通过。')
                print('进行显式多目标优化评估...')
                regen_flag, exp_report = self.evalExpMOs(task_name, user_query,
                                                         target_kw, weights, task_objs_li,
                                                         task_objs_cont,
                                                         exp_li, exp_li_t, gen_cont, r_knowledge_cont)
                self.reports_dic['exp_report'] = exp_report
                if regen_flag == 0:
                    report = exp_report
                    print('显式多目标评估不通过，需要重新生成结果。')
                elif regen_flag == 1:
                    report = exp_report + imp_report
                    print('显式多目标评估不通过，需要重新生成解释。')
                elif regen_flag == 2:
                    report = ''
                    print('显式多目标评估通过。')
                elif regen_flag == 3:
                    report = ''
                    print('显式多目标评估通过，但选择的结果为历史最佳')

        else:
            if self.exp_max_ind != -1:
                report = '超过最大轮次，尝试从历史最佳结果中选取。'
                regen_flag = 3
            else:
                report = '超过最大轮次，异常退出。'
                regen_flag = 4

        return self.integrateEval(report, regen_flag)
