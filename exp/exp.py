from dcclient.dcclient import DataCenterClient
from collections import defaultdict
from bs4 import UnicodeDammit
import numpy as np
import json     
import gensim
import pickle
import networkx as nx
import logging
import pymongo
from sklearn.cluster import Ward, KMeans, MiniBatchKMeans, spectral_clustering

data_center = DataCenterClient("tcp://10.1.1.111:32012")
mongo_client = pymongo.Connection("10.1.1.111",12345)["aminer"]["publications"]
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

authors = []
documents = []
doc_time = {}
doc_author = {}
doc_term = {}
start_time = 0
end_time = 10000
time_window = 1
num_documents = 0
num_documents_given_time = []

num_terms = 0
corpus = []
term_id = {}
term_freq = {}
term_freq_given_time = []
co_occur = {}
co_occur_given_time = []
#co occured pairs
pairs = []
pair_id = {}
num_pairs = 0
mutual_info = {}
mutual_info_given_time = {}

time_slides = []
get_time_slide = {}

G = nx.DiGraph()

def set_time_slides(time_window, start_time, end_time):
    cur = start_time
    while cur < end_time:
        cur_ts = []
        for i in range(time_window):
            cur_ts.append(cur)
            cur += 1
        time_slides.append(cur_ts)
    for i, ts in enumerate(time_slides):
        for y in ts:
            get_time_slide[y] = i

def get_core_community(query, time_window, start_time, end_time):
    print query, time_window, start_time, end_time
    set_time_slides(time_window, start_time, end_time+1)
    authors = []
    for q in query:
        authors.extend(data_center.searchAuthors(q).authors)
    return authors

def get_documents(authors):
    documents = []
    for a in authors:
        logging.info("querying documents for %s from %s to %s" % (a.names, start_time, end_time))
        result = data_center.getPublicationsByAuthorId([a.naid])
        logging.info("found %s documents" % len(result.publications))
        for p in result.publications:
            #update time info
            if p.year >= start_time and p.year <= end_time:
                #insert document
                documents.append(p)
    num_documents = len(documents)
    return documents


def get_terms():
    global num_terms
    for d in documents:
        terms = []
        res = mongo_client.find_one({"_id":d.id})
        tid = None
        for t in res["terms"]:
            if not t in term_id:
                corpus.append(t)
                term_id[t] = num_terms
                num_terms += 1
            tid = term_id[t]
            terms.append(tid)
        doc_term[d.id] = terms
        time = get_time_slide[res["year"]]
        doc_time[d.id] = time
        num_documents_given_time[time] += 1

def calculate_co_occur():
    term_freq = [0 for i in range(num_terms)]
    term_freq_given_time = [[0 for i in range(num_terms)] for i in range(len(time_slides))]
    co_occur = {}
    co_occur_given_time = [{} for i in range(len(time_slides))]
    for d in documents:
        terms = doc_term[d]
        time = doc_time[d]
        for i in range(len(terms)):
            t_i = terms[i]
            term_freq[t_i] += 1
            term_freq_given_time[time][t_i] += 1
            for j in range(i+1, len(terms)):
                t_j = terms[j]
                term_freq[t_j] += 1
                term_freq_given_time[time][t_j] += 1
                #form term pair
                #calculate co occurence
                pair = (t_i, t_j)
                if pair[0] > pair[1]:
                    pair = (pair[1], pair[0])
                if not pair in co_occur:
                    co_occur[pair] = 1
                else:
                    co_occur[pair] += 1
                if not pair in co_occur_given_time[time]:
                    co_occur_given_time[time] = 1
                else:
                    co_occur_given_time[time] += 1
    for pair in co_occur:
        pairs.append(pair)
        pair_id[pair] = num_pairs
        num_pairs += 1
    for i in range(num_terms):
        pairs.append((i,i))
        pair_id[(i,i)] = num_pairs
        num_pairs += 1

def mutual_infomation(u, v, t):
    mi = 0.0
    pair = (u, v)
    if pair[0] > pair[1]:
        pair = (pair[1], pair[0])
    D = float(num_documents_given_time[t])
    df_u = term_freq[u]
    df_v = term_freq[v]
    df_u_v = co_occur_given_time[t][pair]
    df_u_nv = df_u - df_u_v
    df_nu_v = df_v - df_u_v
    df_nu_nv = D - df_u - df_v + df_u_v
    h_1 = (df_u_v/D)*numpy.log((df_u_v*D)/(df_u*df_v))
    h_2 = (df_u_nv/D)*numpy.log((df_u_nv*D)/(df_u*(D-df_v)))
    h_3 = (df_nu_v/D)*numpy.log((df_nu_v*D)/((D-df_u)*df_v))
    h_4 = (df_nu_nv/D)*numpy.log((df_nu_nv*D)/((D-df_u)*(D-df_v)))
    mi = h_1 + h_2 + h_3 + h_4
    return mi

def calculate_mutual_info():
    mutual_info = [0.0 for i in range(num_pairs)]
    mutual_info_given_time = [[0.0 for i in range(num_pairs)] for t in range(len(time_slides))]
    for t in range(len(time_slides)):
        for i, p_i in enumerate(pairs):
            mutual_info_given_time[t][i] = mutual_infomation(p_i[0], p_i[1], t)


def calculate_burstiness():
    init_threshold = 0.5
    beta = 0.8
    burstiness = []
    burstiness.append([init_threshold for i in range(num_terms) ])
    accu_term_frequency = np.zeros(num_terms)
    accu_num_docs = 0
    for ts in range(1,len(time_slides)+1):
        prev_ts = ts - 1
        accu_term_frequency += term_freq_given_time[prev_ts,:]
        accu_num_docs += num_documents_given_time[prev_ts]
        burstiness_ts = [0 for i in range(num_terms)]
        for term in range(num_terms):
            #_ts_term
            a = term_freq_given_time[ts, term]
            #_ts_nterm
            b = num_documents_given_time[ts] - a
            #_nts_term
            c = accu_term_frequency[term]
            #_nts_nterm
            d = accu_num_docs - c

            chi2 = chi_square(a,b,c,d)
            burstiness_ts[i] = burstiness[prev_ts][term] * beta + (1-beta) * math.exp(-chi2)
        burstiness.append(burstiness_ts)
    return burstiness

def chi_square(a,b,c,d):
    n = a+b+c+d
    return (a*d-b*c)**2 * n * 1.0 / ((a+b)*(c+d)*(a+c)*(b+d))


def build_network():
    #assign influence probability
    for t in range(len(time_slides)):
        for i, p_i in enumerate(pairs):
            mi_0_1 = mutual_info_given_time[t][i]
            mi_0_0 = mutual_info_given_time[t][pair_id[(p_i[0], p_i[0])]]
            mi_1_1 = mutual_info_given_time[t][pair_id[(p_i[1], p_i[1])]]
            inf_0_1 = mi_0_1/mi_0_0
            inf_1_0 = mi_0_1/mi_1_1
            G.add_edge(str(p_i[0])+"*"+str(t), str(p_i[1])+"*"+str(t), influence=inf_0_1)
            G.add_edge(str(p_i[1])+"*"+str(t), str(p_i[0])+"*"+str(t), influence=inf_1_0)
        if t > 0:
            for term, i in enumerate(corpus):
                G.add_edge(str(i)+"*"+str(t), str(i)+"*"+str(t-1), influence=1)
                G.add_edge(str(i)+"*"+str(t-1), str(i)+"*"+str(t), influence=1)
    #assign activate threshold


def influence_maximization(G, num_seed, step):
    from linear_threshold import linear_threshold
    linear_threshold(G, num_seed, step)


def trend_partitioning():
    pass


def main():
    authors = get_core_community("data mining", 1, 1960, 2012)


if __name__ == "__main__":
    main()