import argparse
import sys
import os
import time
from pathlib import Path
# from discussion import LLM_Discussion_AUT, LLM_Discussion_Scientific, LLM_Discussion_Instance_Similarities
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
import json
import re
from agents import OpenAIAgent, GeminiAgent, QwenAgent, GLMAgent, MistralAgent, BaichuanAgent, DeepSeekAgent
import datetime
import os
import pandas as pd

# python Other_baselines/LLM-Discussion-main/Experiments/multi_agent_new/llm_discussion_new.py
llm = 'deepseek'  # qwen，gemini，mistral，gpt4o，glm4,glm-4-flash,baichuan,deepseek
s_n = 0
e_n = 100
print(s_n,'~',e_n)
print(f"Running LLM Discussion with {llm} agents.")
# base_os = '/home/zsl/audrey_code/AI_Naming/'
base_os = 'D:/A_MyStudy/AI_Naming/'
# base_os = '/AIOT-vePFS/sf01445601/audrey/AI_Naming/'
f_os = base_os + f'dataset/test_dataset/baselines/0818/baseline_{llm}.csv'
if os.path.exists(f_os):
    df_exit = pd.read_csv(f_os)
# else:
#     # 构建结果表
#     df_exit = pd.DataFrame(columns=['query', 'name', 'exp', 'r_poem', 'backbone', 'method', 'up_w', 'output'])
#     df_exit.to_csv(f_os, index=False, encoding='utf-8', header=True)


class Discussion:
    # PROMPTS = {
    #     1: "You are in a group discussion with other teammates; as a result, answer as diversely and creatively as you can.",
    #     2: "You're in a brainstorming session where each idea leads to the next. Embrace the flow of creativity without limits, encouraging one another to build on each suggestion for unexpected connections.",
    #     3: "Pretend your team is at a think tank where unconventional ideas are the norm. Challenge each other to think from different perspectives, considering the most unusual or innovative ideas.",
    #     4: "Engage in a collaborative discussion where each of you contributes a unique insight or query, aiming to delve into uncharted territories of thought. Throughout the discussion, focus on expanding the scope and depth of each contribution through constructive feedback, counterpoints, and further questioning. The objective is to achieve a broad spectrum of ideas and solutions, promoting a culture of continuous learning and innovation.",
    #     5: "Envision your group as a crew on a mission to solve a mystery using only your creativity and wit. How would you piece together clues from each member's ideas to find the solution? And this would be crucial to your member’s life."
    # }
    PROMPTS = {1: "您正在与其他队友进行小组讨论；因此，请尽可能地以多样化和创造性的方式回答问题。",
               2: "您正在进行头脑风暴会议，每个想法都会引出下一个想法。拥抱无限的创造力，鼓励彼此在每个建议的基础上建立意想不到的联系。",
               3: "假设您的团队是一个智囊团，非常规想法是常态。挑战彼此从不同角度思考，考虑最不寻常或最具创新性的想法。",
               4: "参与协作讨论，每个人都贡献独特的见解或疑问，旨在深入探索未知的思想领域。在整个讨论过程中，通过建设性的反馈、反驳和进一步提问，专注于扩大每个贡献的范围和深度。目标是实现广泛的想法和解决方案，促进持续学习和创新的文化。",
               5: "将您的团队想象成一个团队，仅使用您的创造力和智慧来解开谜团。你会如何从每个成员的想法中拼凑出线索来找到解决方案？这对你的成员的生活至关重要。"}  # we补

    def __init__(self, dataset_file, rounds, prompt):
        self.dataset_file = dataset_file
        self.rounds = rounds
        self.discussion_prompt = self.PROMPTS.get(prompt, "Invalid prompt selected.")
        # print("Discussion initialized with dataset: {} and {} rounds.".format(dataset_file, rounds))

    def run(self):
        pass

    def save_conversation(self, filename, conversation_data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            json.dump(conversation_data, file, indent=4)
        print(f"Saved Conversation Data to {filename}")

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r', encoding="utf-8") as f:
            return json.load(f)

    def extract_response(self, content):  # we补
        # lines = content.split('\n')
        # uses = [line.strip() for line in lines if line.strip() and re.match(r"^\d+\.", line)]
        # uses = [use[use.find('.') + 2:] for use in uses]
        temp_res = content.replace('\n', '').replace(' ', '')
        # 正则表达式模式来匹配名字和解释
        pattern = r'"名字":\s*"([^"]+)",\s*"解释":\s*"([^"]+)"'
        # 使用findall方法提取所有匹配
        matches = re.findall(pattern, temp_res)
        # 将结果转为字典并合并到一个列表中
        uses = [{"名字": name, "解释": explanation} for name, explanation in matches]

        if len(uses) == 0:
            # 提取 JSON 部分
            json_strings = re.findall(r'```json\n(.*?)\n```', content, re.DOTALL)
            for json_str in json_strings:
                try:
                    data = json.loads(json_str)  # 解析 JSON
                    if "名字" in data and "解释" in data:
                        uses.append({"名字": data["名字"], "解释": data["解释"]})
                except json.JSONDecodeError:
                    continue  # 忽略解析失败的部分

        return uses


class LLM_Debate(Discussion):
    def __init__(self, agents_config, dataset_file, rounds, task, prompt):
        super().__init__(dataset_file, rounds, prompt)
        self.task_type = task
        self.agents = self.initialize_agents(agents_config)
        print(f"LLM_Debate initialized for task: {task} with {len(self.agents)} agents.")

    def initialize_agents(self, agents_config):
        agents = []
        for config in agents_config:
            config['type'] = llm
            if config['type'] == 'gpt4o':
                agents.append(OpenAIAgent(model_name=config['model_name'],
                                          agent_name=config['agent_name'],
                                          agent_role=config['agent_role'],
                                          agent_speciality=config['agent_speciality'],
                                          agent_role_prompt=config['agent_role_prompt'],
                                          speaking_rate=config['speaking_rate']))
            elif config['type'] == 'gemini':
                agents.append(GeminiAgent(model_name=config['model_name'],
                                          agent_name=config['agent_name'],
                                          agent_role=config['agent_role'],
                                          agent_speciality=config['agent_speciality'],
                                          agent_role_prompt=config['agent_role_prompt'],
                                          speaking_rate=config['speaking_rate']))
            elif config['type'] == 'qwen':  # we补
                agents.append(QwenAgent(model_name=config['model_name'],
                                        agent_name=config['agent_name'],
                                        agent_role=config['agent_role'],
                                        agent_speciality=config['agent_speciality'],
                                        agent_role_prompt=config['agent_role_prompt'],
                                        speaking_rate=config['speaking_rate']))
            elif config['type'] == 'glm4' or config['type'] == 'glm-4-flash':  # we补
                agents.append(GLMAgent(model_name=config['model_name'],
                                       agent_name=config['agent_name'],
                                       agent_role=config['agent_role'],
                                       agent_speciality=config['agent_speciality'],
                                       agent_role_prompt=config['agent_role_prompt'],
                                       speaking_rate=config['speaking_rate']))
            elif config['type'] == 'mistral':  # we补
                agents.append(MistralAgent(model_name=config['model_name'],
                                           agent_name=config['agent_name'],
                                           agent_role=config['agent_role'],
                                           agent_speciality=config['agent_speciality'],
                                           agent_role_prompt=config['agent_role_prompt'],
                                           speaking_rate=config['speaking_rate']))
            elif config['type'] == 'baichuan':  # we补
                agents.append(BaichuanAgent(model_name=config['model_name'],
                                            agent_name=config['agent_name'],
                                            agent_role=config['agent_role'],
                                            agent_speciality=config['agent_speciality'],
                                            agent_role_prompt=config['agent_role_prompt'],
                                            speaking_rate=config['speaking_rate']))
            elif config['type'] == 'deepseek':  # we补
                agents.append(DeepSeekAgent(model_name=config['model_name'],
                                            agent_name=config['agent_name'],
                                            agent_role=config['agent_role'],
                                            agent_speciality=config['agent_speciality'],
                                            agent_role_prompt=config['agent_role_prompt'],
                                            speaking_rate=config['speaking_rate']))
            else:
                raise ValueError(f"Unsupported agent type: {config['type']}")
        return agents

    def construct_response(self, question, most_recent_responses, current_agent, is_last_round, baseline=False):
        # prefix_string = "These are the solutions to the problem from other agents:\n"
        prefix_string = "以下是其他代理针对该问题提供的解决方案：\n"  # we补
        for agent_name, responses in most_recent_responses.items():
            if agent_name == current_agent.agent_name:
                continue
            if responses and 'parts' in responses[-1]:
                response_content = responses[-1]['parts'][0]
            else:
                response_content = responses[-1]['content']

            # other_agent_response = f"One agent solution: ```{response_content}```\n"
            other_agent_response = f"一个代理的解决方案: ```{response_content}```\n"
            prefix_string += other_agent_response

        if baseline:
            # Using the reasoning from other agents as additional advice, can you give an updated answer? Please put your answer in a list format, starting with 1. ... 2. ... 3. ... and so on.
            # suffix_string = "参考其它代理的推理结果作为补充建议，您能否给出更新的答案？请以列表形式列出您的答案，从 1. ... 2. ... 3. ... 等开始。"  # we补
            suffix_string = "参考其它代理的推理结果作为补充建议，您能否给出更新的答案？请仍以JSON形式返回您的答案，即{'名字':'...', '解释':'...'}，若有多个答案则按照相同的格式列出多个结果。"  # we补
        else:
            suffix_string = question + self.discussion_prompt
            if is_last_round:
                # suffix_string += " This is the last round of the discussion, please finalize and present a list of as many creative answers as possible. Please list the final response in 1. ... 2. ... 3. ... and so on. \n\n"
                suffix_string += "这是讨论的最后一轮，请最终确定并列出尽可能多的有创意的答案。请仍以JSON形式返回您的答案，即{'名字':'...', '解释':'...'}，若有多个答案则按照相同的格式列出多个结果。 \n\n"  # we补

        return prefix_string + suffix_string

    def save_debate_conversations(self, agents, all_responses, init_results, final_results, amount_of_data,
                                  task_type="AUT", baseline=False):
        current_date, formatted_time = self.get_current_datetime()
        model_names_concatenated = self.concatenate_model_names(agents)
        role_names_concatenated = self.concatenate_role_names(agents)
        subtask = self.determine_subtask(agents, baseline)

        output_filename = self.generate_filename(task_type, subtask, "chat_log", model_names_concatenated,
                                                 role_names_concatenated, current_date, formatted_time, amount_of_data,
                                                 len(agents), self.rounds)
        final_ans_filename = self.generate_final_filename(task_type, subtask, "multi_agent", model_names_concatenated,
                                                          role_names_concatenated, current_date, formatted_time,
                                                          amount_of_data, len(agents), self.rounds)
        init_ans_filename = self.generate_filename(task_type, subtask, "init", model_names_concatenated,
                                                   role_names_concatenated, current_date, formatted_time,
                                                   amount_of_data, len(agents), self.rounds)

        # Ensure all required directories exist
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        os.makedirs(os.path.dirname(final_ans_filename), exist_ok=True)
        os.makedirs(os.path.dirname(init_ans_filename), exist_ok=True)

        self.save_conversation(output_filename, all_responses)
        self.save_conversation(final_ans_filename, final_results)
        self.save_conversation(init_ans_filename, init_results)

        return final_ans_filename

    @staticmethod
    def get_current_datetime():
        current_time = datetime.datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        formatted_time = current_time.strftime("%H-%M-%S")
        return current_date, formatted_time

    @staticmethod
    def concatenate_model_names(agents):
        if all(agent.model_name == agents[0].model_name for agent in agents):
            return agents[0].model_name.replace(".", "-")
        return "-".join(agent.model_name.replace(".", "-") for agent in agents)

    @staticmethod
    def concatenate_role_names(agents):
        if all(agent.agent_role == "None" for agent in agents):
            return "None"
        return "-".join(agent.agent_role.replace(" ", "") for agent in agents)

    def determine_subtask(self, agents, baseline):
        if baseline:
            return "baseline"
        if all(agent.agent_role == "None" for agent in agents):
            return "FINAL"
        return "roleplay"

    @staticmethod
    def generate_filename(task_type, subtask, data_type, model_names_concatenated, role_names_concatenated,
                          current_date, formatted_time, amount_of_data, num_agents, num_rounds):
        return f"../../Results/{task_type}/{data_type}/{task_type}_multi_debate_{subtask}_{num_agents}_{num_rounds}_{model_names_concatenated}_{role_names_concatenated}_{data_type}_{current_date}-{formatted_time}_{amount_of_data}.json"

    @staticmethod
    def generate_final_filename(task_type, subtask, data_type, model_names_concatenated, role_names_concatenated,
                                current_date, formatted_time, amount_of_data, num_agents, num_rounds):
        return f"../../Results/{task_type}/Output/{data_type}/{task_type}_multi_debate_{subtask}_{num_agents}_{num_rounds}_{model_names_concatenated}_{role_names_concatenated}_{data_type}_{current_date}-{formatted_time}_{amount_of_data}.json"


def saveToCSV(final_result, up_w):  # we补
    queries = []
    anw_1 = []
    anw_2 = []
    name_1 = []
    name_2 = []
    final_ans = []
    for res in final_result:
        q = res['question'].split('\n')[1][6:]
        if q not in queries:
            queries.append(q)
        if res['Agent'] == 'Agent 1 - 古诗词文化专家':
            anw_1 = res['answer']
            name_1 = [j['名字'] for j in anw_1]
        elif res['Agent'] == 'Agent 2 - 风水命理专家':
            anw_2 = res['answer']
            name_2 = [j['名字'] for j in anw_2]
        elif res['Agent'] == 'Agent 3 - 现代汉语言学家':
            anw_3 = res['answer']
            name_3 = [j['名字'] for j in anw_3]
            name_ = list(set(name_1) & set(name_2) & set(name_3))
            if len(name_) == 0:
                name_ = list(set(name_1) & set(name_2))
            if len(name_) == 0:
                name_ = list(set(name_1) & set(name_3))
            if len(name_) == 0:
                name_ = list(set(name_2) & set(name_3))
            if len(name_) == 0:
                continue
                #   name_ = list(set(name_1) | set(name_2) | set(name_3))
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
            final_ans.append({'query': q, 'name': max_name, 'exp': max_exp, 'r_poem': None, 'backbone': llm,
                              'method': 'llm_discussion', 'up_w': up_w, 'output': None})
    df = pd.DataFrame(final_ans)
    # 将df增加到csv文件中
    df.to_csv(f_os, mode='a', header=False, index=False)


class LLM_Discussion_Instance_Similarities(LLM_Debate):
    def run(self):
        # with open(self.dataset_file, 'r') as f:
        #     dataset = json.load(f)
        f = base_os + 'dataset/test_dataset/benchmark/test_data_500.csv'
        df = pd.read_csv(f)
        q_li = df['query'].tolist()[s_n:e_n]
        df['query'] = df['query'].apply(lambda
                                            x: '请结合用户需求和任务目标取一个合适的名字，尽可能满足所有目标，最终按照输出说明给出结果。\n-用户需求：' + x + '\n-任务目标：文化内涵（古诗）、父母期待、五行八字、个人特征（性别、生肖、出生年月等）、其他需求。\n输出说明(JSON格式)：{"名字":"...", "解释":"..."}\n注意：将每条推荐结果都整合为JSON格式！！')

        # num = df.shape[0]
        dataset = {'Examples': df['query'].tolist()[s_n:e_n]}
        up_w_li = df['up_w'].tolist()[s_n:e_n]
        all_responses = {}
        init_results = []
        final_results = []
        amount_of_data = len(dataset['Examples'])
        ind = 0
        for example in tqdm(dataset['Examples']):
            df_res = df_exit[(df_exit['query'] == q_li[ind]) & (df_exit['method'] == 'llm_discussion')]
            if df_res.shape[0] > 0:  # 如果已存在，则跳过
                ind += 1
                time.sleep(0.01)
                print(f"Skip {ind}!")
                continue
            try:
                chat_history = {agent.agent_name: [] for agent in self.agents}
                # print("initial chat_history: ", chat_history, "\n")
                # --------------->>>> set the system content
                question = example
                initial_prompt = "与他人发起讨论，共同完成以下任务：" + question + self.discussion_prompt
                # ------------------------------------------
                most_recent_responses = {}
                temp_final_results = []  # we补
                for round in tqdm(range(self.rounds)):
                    is_last_round = (round == self.rounds - 1)
                    is_first_round = (round == 0)
                    round_responses = {agent.agent_name: [] for agent in self.agents}
                    # print(f"Round {round + 1}: Discussion on {question}")
                    for agent in self.agents:

                        if agent.agent_role != "None":
                            agent_role_prompt = f"您是{agent.agent_role}，擅长的领域是{agent.agent_speciality}。{agent.agent_role_prompt}请记住在每次对话开始时声明您的角色。 "
                            # print(f"agent_role = {agent.agent_role}")
                        else:
                            agent_role_prompt = ""

                        if is_first_round:
                            formatted_initial_prompt = agent.construct_user_message(agent_role_prompt + initial_prompt)
                            chat_history[agent.agent_name].append(formatted_initial_prompt)
                            response = agent.generate_answer(chat_history[agent.agent_name])
                            response_list = self.extract_response(response)  # we补
                            init_result = {"question": question, "answer": response_list, "Agent": agent.agent_name}
                            init_results.append(init_result)
                        else:
                            combined_prompt = self.construct_response(question, most_recent_responses, agent,
                                                                      is_last_round)
                            formatted_combined_prompt = agent.construct_user_message(
                                agent_role_prompt + combined_prompt)  # 角色设定+问题+上一次讨论的其他的agents的回答
                            chat_history[agent.agent_name].append(formatted_combined_prompt)
                            response = agent.generate_answer(chat_history[agent.agent_name])  # 该角色依据历史生成记录，生成新的回答

                            # Save Final Result
                            if is_last_round:
                                response_list = self.extract_response(response)
                                # print(f"response_list = {response_list}")
                                final_result = {"question": question, "answer": response_list,
                                                "Agent": agent.agent_name}
                                final_results.append(final_result)
                                temp_final_results.append(final_result)

                        formatted_response = agent.construct_assistant_message(response)
                        chat_history[agent.agent_name].append(formatted_response)  # Update the agent's chat history
                        round_responses[agent.agent_name].append(formatted_response)
                    most_recent_responses = round_responses
                saveToCSV(temp_final_results, up_w_li[ind])  # we补
                print(f"Saved successfully {ind}!")  # we补
                ind += 1  # we补
                all_responses[question] = chat_history
            except:
                print("Error in running the discussion on the question: ", example)
                continue
        output_file = self.save_debate_conversations(self.agents, all_responses, init_results, final_results,
                                                     amount_of_data, task_type=self.task_type)
        return output_file


# This file run LLM Discussion

def main():
    parser = argparse.ArgumentParser(description="协调与多个AI代理的讨论。")
    args = parser.parse_args()

    args.config = base_os + 'Other_baselines/LLM-Discussion-main/Experiments/multi_agent_new/config_role.json'  # 我们改的配置文件
    args.dataset = base_os + 'Other_baselines/LLM-Discussion-main/Datasets/Similarities/similarities_3.json'  # 后面已经改成我们的数据集，这里没有用
    args.rounds = 5  # 5轮讨论
    args.type = 'Similarities'
    args.eval_mode = False
    args.prompt = 5  # 选择1-5的discuss提示语

    agents_config = LLM_Discussion_Instance_Similarities.load_config(args.config)
    discussion_runner = LLM_Discussion_Instance_Similarities(agents_config, args.dataset, args.rounds, args.type,
                                                             args.prompt)
    discussion_output = discussion_runner.run()


if __name__ == "__main__":
    main()

