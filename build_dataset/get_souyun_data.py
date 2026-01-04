'''
从搜韵网爬取数据
https://www.sou-yun.cn/AllusionsIndex.aspx
'''

# 从https://www.sou-yun.cn/AllusionsIndex.aspx这个网站爬取数据
from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm
import pandas as pd
from utils.base_param import BaseParam
import requests
import re

params = BaseParam()
web_data_os = params.web_data_os
# 这些始终要用到，作为全局，可以加快code运行速度
chrome_options = webdriver.ChromeOptions()
# 使用headless无界面浏览器模式
chrome_options.add_argument('--headless')  # 增加无界面选项
chrome_options.add_argument('--disable-gpu')  # 如果不加这个选项，有时定位会出现问题
# 1\启动浏览器
driver = webdriver.Chrome(options=chrome_options)
# 隐式等待
driver.implicitly_wait(10)
word_base_link = 'https://www.sou-yun.cn/AllusionsIndex.aspx'
base_link = 'https://www.sou-yun.cn/'
type_url = 'https://www.sou-yun.cn/QueryPoem.aspx'



def get_soup2(url):  # 给一个链接返回封装之后的结果
    # headers = {'User-Agent': random.choice(agent_list)}
    # time.sleep(random.uniform(0.5, 1))
    try:
        response = requests.get(url, timeout=5)  # 使用代理
    except:
        return None
    content = response.text
    soup = BeautifulSoup(content, 'lxml')
    return soup


def get_soup(url):  # 给一个链接返回封装之后的结果
    # time.sleep(random.uniform(0.5, 1))
    driver.get(url)
    content = driver.page_source
    soup = BeautifulSoup(content, 'lxml')
    return soup


def get_souyun_data(f_words):  # 获取搜韵的keywords_exp
    words_all = []
    related_words_all = []
    word_meanings_all = []
    for i in tqdm(range(103)):
        page = get_soup(word_base_link + '?page=' + str(i))
        table = page.find('div', id='main').find('div', 'wrapper').find('table')
        words = []
        nums = []
        for i in table.tr.td.find_all('a'):
            words.append(i.text.split('（')[0])
            nums.append(i.get('href'))
        related_words = []
        word_meanings = []
        main_con = table.find_all('div', 'allusion')
        for ind in range(len(main_con)):
            num = nums[ind].split('(')[1].split(')')[0]
            rw = main_con[ind].find('div', id='allusion_head_' + num).find_all('div', 'inline')
            rw = [i.text for i in rw]
            related_words.append(rw)
            try:
                word_meanings.append(main_con[ind].find_all('div', 'allusionBlock')[2].p.text)
            except:
                word_meanings.append(None)
        words_all.extend(words)
        related_words_all.extend(related_words)
        word_meanings_all.extend(word_meanings)
    data = pd.DataFrame({'word': words_all, 'related_words': related_words_all, 'word_meanings': word_meanings_all})
    data.to_csv(f_words, index=False)


def write_souyun_topics(f_topics):  # 先获取所有的tags—url
    t_soup = get_soup(type_url)
    all_topics_code = t_soup.find('div', 'full').find_all('a')  # 所有的主题code
    topics_code = t_soup.find('div', 'full').find_all('p')  # 所有的大主题code
    topics = []  # 所有的大主题文本
    for i in topics_code:
        topics.append(i.text.strip())
    sub_topics = []  # 小主题
    url_topics = []  # 小主题链接,所属的大主题
    c = -1
    for j in all_topics_code:
        con = j.text.strip()
        if con in topics[c + 1:]:  # 防止小主题里面有与大主题一样的题目
            c += 1
        else:
            sub_topics.append(con)
            url_topics.append([j.get('href'), topics[c]])
    with open(f_topics, 'w+', encoding='utf-8') as f:
        for i in zip(sub_topics, url_topics):
            f.write(str(i) + '\n')


def trans_con(cont):
    clean_c1 = re.sub(r'[(（][^)）]*[)）]', '', cont)  # 去除括号及括号内的内容
    clean_c2 = re.sub(r'[，。？！；：、]', '|', clean_c1)  # 将标点符号替换为竖线
    clean_c2 = re.sub(r'<[^>]+>', '', clean_c2)  # 去除html标签
    new_cont = (clean_c2.replace('\n', '').replace('\r', '')
                .replace(' ', '').replace('||', '|'))
    new_cont = keep_chinese_and_pipe(new_cont)  # 仅保留中文和|字符
    return new_cont


def keep_chinese_and_pipe(text):
    # 使用正则表达式匹配中文字符和竖线 |
    pattern = re.compile(r'[^\u4e00-\u9fa5|]')
    result = pattern.sub('', text)
    return result


def get_souyun_tags(f_input, f_topics, f_tag):  # 从所有主题链接中获取诗词，对照现有诗词库，更新tags
    # n = 3
    # print(n)
    # start_n = 0 + 10 * n
    # end_n = min(10 + 10 * n, 729)
    cnt = 0
    start_n = 0
    end_n = 729
    sub_topic = []
    url_topic = []
    df_poems = pd.read_csv(f_input, encoding='utf-8', low_memory=False)
    with open(f_topics, 'r', encoding='utf-8') as f:
        for i in f:
            sub_topic.append(eval(i)[0])
            url_topic.append(eval(i)[1])
    end_n = len(sub_topic)
    for k, v in tqdm(zip(sub_topic[start_n:end_n], url_topic[start_n:end_n]),
                     total=end_n - start_n + 1):  # k:小主题，v:[链接，大主题]
        cnt += 1
        tem_soup = get_soup(base_link + v[0])
        try:
            titles = tem_soup.find('div', 'full').find_all('a')
        except:
            continue
        poems = {}
        for i in tqdm(titles):
            title = i.text.strip()
            poems[title] = i.get('href')
            try:
                # 从df找到对应的诗词名字、内容，然后找到对应的tag值，然后更新tag值
                poem_soup = get_soup2(base_link + i.get('href'))
                content = poem_soup.find('div', 'poemContent').text
                # 对爬取的content进行标准化处理
                con = trans_con(content)[:-1]
                # 用title、content匹配
                # s_re = df_poems.loc[df_poems['title'] == title].loc[df_poems['content'].str.match('^' + con), 'tags']
                s_re = df_poems.loc[df_poems['content'] == con, 'tags']
                # print(title, con, s_re)
                tmp_ind = s_re.index[0]
                tmp = eval(s_re.values[0])
                if tmp['topics'] is None:
                    tmp['topics'] = v[1] + ',' + k + ';'
                elif v[1] in tmp['topics']:
                    tmp['topics'] += k + ';'  # 用;分隔
                else:
                    tmp['topics'] += v[1] + ',' + k + ';'
                df_poems.at[tmp_ind, 'tags'] = tmp
            except:
                continue
        if cnt % 10 == 0:
            df_poems.to_csv(f_tag, index=False)
    df_poems.to_csv(f_tag, index=False)


if __name__ == '__main__':
    file_words = "souyun_words.csv"
    file_topics = 'souyun_sub_topics_url.txt'
    file_input = 'poems_BaW.csv'
    file_tag = 'poems_BaW_tags.csv'
    # # 获取搜韵的keywords_exp
    # get_souyun_data(web_data_os + file_words)
    # # 获取搜韵的topics
    # write_souyun_topics(web_data_os + file_topics)
    # 获取搜韵的tags
    get_souyun_tags(params.inter_data_os + file_input, web_data_os + file_topics, web_data_os + file_tag)
