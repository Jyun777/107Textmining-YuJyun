import os #開啟資料
import pickle #資料形式
import jieba #斷字函數
import operator 
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager #字型管理
from datetime import datetime #時間函數
from collections import Counter #計算字數

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator #文字雲
from PIL import Image #圖片

font_path = '../tools/msjh.ttc' #定義字型
font = font_manager.FontProperties(fname='../tools/msjh.ttc',
                                   weight='bold',
                                   style='normal', size=16)


# jieba.set_dictionary('../jieba_data/dict.txt.big')
# jieba.load_userdict('../jieba_data/userdict.txt')
stopwords = [] #讀入stopwords
with open('../jieba_data/stopwords.txt', 'r', encoding='UTF-8') as file: #encoding='UTF-8' 爬蟲資料可讀取中文字
    for each in file.readlines():
        stopwords.append(each.strip())
    stopwords.append(' ')

# with open('crawler/data/new_talk.pkl', 'rb') as f:
#     data = pickle.load(f)
    
# data = data[::-1]
# contents = [news['content'] for news in data]

# date_list = [news['date'] for news in data]
# all_date = sorted(list(set(date_list)))
# aall_date = [date[5:] for date in all_date][::-1]
# date_index = [date_list.index(each_date) for each_date in all_date]
# date_index.append(len(date_list)-1)
# number_of_news = [date_index[i+1] - date_index[i]-1 for i in range(len(date_index)-1)]
# number_of_terms = [sum([sum(data[ni]['cutted_dict'].values()) for ni in range(date_index[i], date_index[i+1])]) for i in range(len(date_index)-1)]


def remove_punctuation(content_string, user_pc=False): #移除標點符號 1. 有亂碼產生; 2. 標點符號不計做詞
    if(user_pc): #若使用者無自訂標點符號，則為函數內定
        punctuation = user_pc 
    else:
        punctuation=list("!@#$%^&*()_+=-[]`~'\"|/\\abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.;{}\r\xa0\u3000、，。「」！？；：<>")
        
    for p in punctuation:
        content_string = content_string.replace(p, " ") #用space替代標點符號
    return(content_string)

def remove_stopwords_from_dict(word_dict, stopwords):
    for w in stopwords:
        word_dict.pop(w, word_dict)
    return word_dict

def lcut_to_dict(lcut):
    word_dict = dict(Counter(lcut))
#     word_dict.pop(' ')
    return(remove_stopwords_from_dict(word_dict, stopwords))

def sort_dict_by_values(d):
    return(sorted(d.items(), key=lambda x: x[1], reverse=True))

def news_containing_keyword(keyword, news_list):
    return list(filter(lambda news: keyword in news, news_list))

def data_containing_keyword(keyword, data):
    return list(filter(lambda news: keyword in news['cutted_dict'].keys(), data))

def news_containing_keywords(keywords, news_list):
    news = news_list
    for keyword in keywords:
        news = news_containing_keyword(keyword, news)
        
    return news

def get_coshow(contents): #從每篇內容斷句截詞出來
    coshow_dict = {}
    cat_content = ' '.join(contents)
    clean_content = remove_punctuation(cat_content) #移除標點符號
    cut_content = jieba.lcut(clean_content)
    cut_content = list(filter(lambda x: x!=' ', cut_content))
    for i in range(len(cut_content)-1):
        wcw = cut_content[i] + cut_content[i+1] #將兩個相鄰的詞也定義成一個詞彙
    #     print(wcw)
        try:
            coshow_dict[wcw] = coshow_dict[wcw] + 1 #計算出現幾次
        except:
            coshow_dict[wcw] = 1

    sdbv = sort_dict_by_values(coshow_dict) #依出現的詞數排列
    return sdbv

def get_cutted_dict(list_of_news):
    cat = ' '.join(list_of_news) #將每篇新聞一起看
    cat = remove_punctuation(cat)
    cutted = jieba.lcut(cat)
    return lcut_to_dict(cutted)

def first_n_words(cutted_dict, n, word_len=2, to=1000):
    sdbv = sort_dict_by_values(cutted_dict)
    return list(filter(lambda x: len(x[0])>=word_len and len(x[0])<=to, sdbv))[:n]

def get_wordcloud_of_keywords(keywords, list_of_news, image_path=False):
    if type(keywords) == str:
        keywords = [keywords]
    
    if image_path:
        coloring = np.array(Image.open(os.path.join(image_path))) #儲存圖片的顏色
        color_func = ImageColorGenerator(coloring) #產生相應顏色碼
        wc = WordCloud(max_font_size=30,
                       background_color="white",
                       mask=coloring,
                       color_func=color_func,
                       font_path=font_path,
                       width=1000, height=1000,
                      max_words=10000)
    else: #若無給圖片則以下列方式儲存
        wc = WordCloud(max_font_size=30,
                       background_color="white",
                       colormap='Set2',
                       font_path=font_path,
                       width=1000, height=300,
                      max_words=1000)
    
    keyword_news = news_containing_keywords(keywords, list_of_news)
    keyword_dict = get_cutted_dict(keyword_news)
    print(len(keyword_dict))
    im = wc.generate_from_frequencies(keyword_dict)
    return im

def merge_one_day_news_dict(one_day_dict, count='wt', divide = 1):
    all_words = set([word for each_dict in one_day_dict for word in each_dict])
    one_day_wf = {}
    for word in all_words:
        one_day_wf[word] = 0
        for news in one_day_dict:
            if count == 'wt':
                one_day_wf[word] += news.get(word, 0)/divide
            if count == 'occur':
                one_day_wf[word] += bool(news.get(word, 0))/divide
    
    return one_day_wf

def plot_line_of_word(word, date_from='2018-06-07', date_to='2019-01-22'):
    from_index = df.columns.get_loc(date_from)
    to_index = df.columns.get_loc(date_to)+1
    date_length = to_index-from_index
    date_int = date_length//25
    font = font_manager.FontProperties(fname='msjh.ttc',
                                   weight='bold',
                                   style='normal', size=16)
    
    plt.plot(aall_date[from_index:to_index], df.loc[word][date_from:date_to], '-o', label=word)
    plt.legend(prop=font)
    plt.xticks(list(range(0, date_length, date_int)), [aall_date[from_index:to_index][i] for i in range(0, date_length, date_int)])
    
def plot_tfdf_of_word(word, df_tf, df_occur, date_from='2018-06-07', date_to='2019-01-22'):
    from_index = df_tf.columns.get_loc(date_from)
    to_index = df_tf.columns.get_loc(date_to)+1
    date_length = to_index-from_index
    date_int = date_length//25
    font = font_manager.FontProperties(fname='msjh.ttc',
                                   weight='bold',
                                   style='normal', size=16)
    
    plt.plot(aall_date[from_index:to_index], 
             df_tf.loc[word][date_from:date_to]*df_occur.loc[word][date_from:date_to], '-o', label=word)
    plt.legend(prop=font)
    plt.xticks(list(range(0, date_length, date_int)), [aall_date[from_index:to_index][i] for i in range(0, date_length, date_int)])

def get_tfdf(word, df_tf, df_occur):
    tfdf = df_occur.loc[word] * df_tf.loc[word]
    return tfdf

def get_high_tfdf_date(word, df_tf, df_occur):
    tfdf = get_tfdf(word, df_tf, df_occur)
    m = statistics.mean(tfdf)
    s = statistics.stdev(tfdf)
    tfdf_bool = [x > m+s for x in tfdf]
    tfdf_date = {all_date[x[0]]: tfdf[x[0]] for x in list(filter(lambda e: e[1], enumerate(tfdf_bool)))}
    return tfdf_date

def keyword_with_event(keyword):
    news_containing_key = news_containing_keyword(keyword, contents)
    key_dict = get_cutted_dict(news_containing_key)
    key_term = first_n_words(key_dict, 300)
    return list(filter(lambda x: x in hot4, [x[0] for x in key_term]))

def draw_event(event, i, df_tf, df_occur, all_date):
    event_date = get_high_tfdf_date(event, df_tf, df_occur)
    date_index = [all_date.index(x) for x in event_date.keys()]
    plt.scatter(date_index, [i for x in date_index], s=[x*100000 for x in list(event_date.values())])
    
def draw_by_list(tf_list, i):
    plt.scatter(aall_date, [i for x in aall_date], s=[x*10 for x in tf_list])