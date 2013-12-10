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
import re
from sklearn.cluster import Ward, KMeans, MiniBatchKMeans, spectral_clustering

data_center = DataCenterClient("tcp://10.1.1.111:32012")
mongo_client = pymongo.Connection("10.1.1.111",12345)["aminer"]["pub"]
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

exp_title = ""

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
    global time_slides
    global get_time_slide
    time_slides = []
    get_time_slide = {}
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

def get_documents(authors, start_time, end_time):
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

def get_terms_wiki():
    global num_terms
    global num_documents_given_time
    num_documents_given_time = [0 for i in range(len(time_slides))]
    for d in documents:
        terms = []
        res = mongo_client.find_one({"_id":d.id})
        if res == None:
            res = {"wiki_id":[]}
        tid = None
        if "wiki_id" not in res:
            print d.id
        else:
            for t in res["wiki_id"]:
                # at least bigram
                if " " in corpus[t]:
                    #reg = r"(^.*\s?is$)|(is\s.*?$)"
                    reg = r"|".join(["(^.*\s%s$)|(^%s\s.*$)"%(x,x) for x in ["in","is","are","the","a","been","but","was","be","a","there","this","that","to","of","not","so","we","with","than","for","and","wa","it","almost","an","al"]])
                    #reg = r"((^|\s)is(\s|$))|((^|\s)are(\s|$))|((^|\s)the(\s|$))|((^|\s)a(\s|$))|((^|\s)been(\s|$))|((^|\s)but(\s|$))|((^|\s)was(\s|$))|((^|\s)be(\s|$))|((^|\s)a(\s|$))|((^|\s)there(\s|$))|((^|\s)this(\s|$))|((^|\s)that(\s|$))|((^|\s)to(\s|$))|((^|\s)of(\s|$))|((^|\s)not(\s|$))|((^|\s)so(\s|$))|((^|\s)we(\s|$))|((^|\s)with(\s|$))|((^|\s)a(\s|$))of"
                    rule = re.compile(reg)
                    if rule.match(corpus[t]) is not None:
                        continue
                    terms.append(t)
                terms = list(set(terms))
        doc_term[d.id] = terms
        time = get_time_slide[d.year]
        doc_time[d.id] = time
        num_documents_given_time[time] += 1

def get_terms_extracted():
    global num_terms
    global num_documents_given_time
    num_documents_given_time = [0 for i in range(len(time_slides))]
    for d in documents:
        terms = []
        res = mongo_client.find_one({"_id":d.id})
        if res == None:
            res = {"terms":[]}
        tid = None
        for t in res["terms"]:
            if not t in term_id:
                corpus.append(t)
                term_id[t] = num_terms
                num_terms += 1
            tid = term_id[t]
            terms.append(tid)
        doc_term[d.id] = terms
        time = get_time_slide[d.year]
        doc_time[d.id] = time
        num_documents_given_time[time] += 1
   

def get_choosen_terms(k):
    sorted_terms = sorted(range(len(corpus)), key=lambda x: term_freq[x], reverse=True)
    return sorted_terms[:k]

def calculate_co_occur():
    global num_pairs
    global pairs
    global term_freq
    global term_freq_given_time
    global co_occur
    global co_occur_given_time
    num_pairs = 0
    pairs = []
    term_freq = [0 for i in range(num_terms)]
    term_freq_given_time = [[0 for i in range(num_terms)] for i in range(len(time_slides))]
    co_occur = {}
    co_occur_given_time = [{} for i in range(len(time_slides))]
    for d in documents:
        terms = doc_term[d.id]
        time = doc_time[d.id]
        for i in range(len(terms)):
            t_i = terms[i]
            term_freq[t_i] += 1
            term_freq_given_time[time][t_i] += 1
            for j in range(i, len(terms)):
                t_j = terms[j]
                # term_freq[t_j] += 1
                # term_freq_given_time[time][t_j] += 1
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
                    co_occur_given_time[time][pair] = 1
                else:
                    co_occur_given_time[time][pair] += 1
    for pair in co_occur:
        pairs.append(pair)
        pair_id[pair] = num_pairs
        num_pairs += 1
    # for i in range(num_terms):
    #     pairs.append((i,i))
    #     pair_id[(i,i)] = num_pairs
    #     num_pairs += 1

def mutual_infomation(u, v, t):
    mi = 0.0
    pair = (u, v)
    if pair[0] > pair[1]:
        pair = (pair[1], pair[0])
    D = float(num_documents_given_time[t])
    df_u = term_freq_given_time[t][u]
    df_v = term_freq_given_time[t][v]
    try:
        df_u_v = co_occur_given_time[t][pair]
    except:
        df_u_v = 0.0
    df_u_nv = df_u - df_u_v
    df_nu_v = df_v - df_u_v
    df_nu_nv = D - df_u - df_v + df_u_v
    h_1 = (df_u_v/D)*np.log((df_u_v*D)/(df_u*df_v)) if (df_u*df_v) != 0 and df_u_v !=0 else 0.0
    h_2 = (df_u_nv/D)*np.log((df_u_nv*D)/(df_u*(D-df_v))) if (df_u*(D-df_v)) != 0 and df_u_nv != 0 else 0.0
    h_3 = (df_nu_v/D)*np.log((df_nu_v*D)/((D-df_u)*df_v)) if ((D-df_u)*df_v) != 0 and df_nu_v != 0 else 0.0
    h_4 = (df_nu_nv/D)*np.log((df_nu_nv*D)/((D-df_u)*(D-df_v))) if ((D-df_u)*(D-df_v)) != 0 and df_nu_nv != 0 else 0.0
    mi = h_1 + h_2 + h_3 + h_4
    # print df_u,df_v,df_u_v,h_1,h_2,h_3,h_4
    return mi

def calculate_mutual_info():
    global mutual_info
    global mutual_info_given_time
    mutual_info = [0.0 for i in range(num_pairs)]
    mutual_info_given_time = [[0.0 for i in range(num_pairs)] for t in range(len(time_slides))]
    for t in range(len(time_slides)):
        for p in co_occur_given_time[t]:
            mutual_info_given_time[t][pair_id[p]] = mutual_infomation(p[0], p[1], t)

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


def build_network(num_choosen_terms):
    G = nx.DiGraph()
    choosen_terms = set(get_choosen_terms(num_choosen_terms))
    #assign influence probability
    for t in range(len(time_slides)):
        for i, p_i in enumerate(pairs):
            if not p_i[0] in choosen_terms or not p_i[1] in choosen_terms:
                continue
            mi_0_1 = mutual_info_given_time[t][i]
            if mi_0_1 == 0:
                continue
            mi_0_0 = mutual_info_given_time[t][pair_id[(p_i[0], p_i[0])]]
            mi_1_1 = mutual_info_given_time[t][pair_id[(p_i[1], p_i[1])]]
            inf_0_1 = mi_0_1/mi_0_0
            inf_1_0 = mi_0_1/mi_1_1
            G.add_edge(str(p_i[0])+"*"+str(t), str(p_i[1])+"*"+str(t), influence=inf_0_1)
            G.add_edge(str(p_i[1])+"*"+str(t), str(p_i[0])+"*"+str(t), influence=inf_1_0)
        if t > 0:
            for term, i in enumerate(choosen_terms):
                G.add_edge(str(i)+"*"+str(t), str(i)+"*"+str(t-1), influence=0.8)
                G.add_edge(str(i)+"*"+str(t-1), str(i)+"*"+str(t), influence=0.8)
    #assign activate threshold
    return G


def influence_maximization(G, seeds, step):
    from linear_threshold import linear_threshold
    linear_threshold(G, seeds, step)


def trend_partitioning():
    pass

def dump_graph(q, time_window, inf):
    nodes = []
    node_id = {}
    num_nodes = 0
    for n in G.nodes():
        nodes.append(n)
        node_id[n] = num_nodes
        num_nodes += 1
    with open("term_network_%s.txt" % exp_title,"w") as f_out:
        f_out.write("%s\n" % len(nodes))
        for n in G.nodes():
            out_edges = G.edge[n]
            line = "\t".join([(str(node_id[e])+"\t"+str(out_edges[e]["influence"])) for e in out_edges.keys()])
            f_out.write("%s\t%s\n" % (len(out_edges), line))
    with open("term_id_%s.txt" % exp_title,"w") as f_out:
        for n in nodes:
            x = n.split("*")
            f_out.write("%s\t%s\t%s\t%s\t%s\n" % (corpus[int(x[0])], x[1], term_freq_given_time[int(x[1])][int(x[0])], G.degree(n), n))
    return nodes, node_id, num_nodes

def select_term_by_freq(k):
    sorted_terms = sorted(range(len(corpus)), key=lambda x: term_freq[x], reverse=True)
    with open("baseline_freq_%s" % exp_title,"w") as f_out:
        [f_out.write("%s\t%s\n" % (corpus[x],term_freq[x])) for x in sorted_terms[:k]] 


def select_term_by_freq_given_time(nodes, k):
    sorted_nodes = sorted(nodes, key=lambda x: term_freq_given_time[int(x.split("*")[1])][int(x.split("*")[0])]/float(num_documents_given_time[int(x.split("*")[1])]+1), reverse=True)
    with open("baseline_freq_given_time_%s" % exp_title,"w") as f_out:
        [f_out.write("%s\t%s\t%s\t%s\n" % (corpus[int(x.split("*")[0])],x,term_freq[int(x.split("*")[0])],term_freq_given_time[int(x.split("*")[1])][int(x.split("*")[0])])) for x in sorted_nodes[:k]]        

def select_term_by_page_rank():
    pass

def select_term_by_degree(k):
    sorted_node = sorted(G.nodes(), key=lambda x:G.degree(x), reverse=True)
    with open("baseline_degree_%s" % exp_title,"w") as f_out:
        [f_out.write("%s\t%s\t%s\n" % (corpus[int(x.split("*")[0])],x,G.degree(x))) for x in sorted_node[:k]]

def load_data():
    global corpus
    global term_id
    global num_terms
    with open("word2id.pickle","rb") as f_in:
        term_id = pickle.load(f_in) 
    with open("corpus.pickle","rb") as f_in:
        corpus = pickle.load(f_in)
    print "load finished"
    num_terms = len(corpus)

def run(q, time_window, start_time, end_time, num_choosen_terms, smooth):
    global exp_title 
    global authors
    global documents
    global G
    exp_title = "%s-%s-%s-%s-%s-%s"%(q, time_window, start_time, end_time, num_choosen_terms, smooth)
    authors = get_core_community([q], time_window, start_time, end_time)
    documents = get_documents(authors, start_time, end_time)
    get_terms_wiki()
    calculate_co_occur()
    calculate_mutual_info()
    G = build_network(num_choosen_terms)
    nodes, node_id, num_nodes = dump_graph(q, time_window, smooth)
    k = num_choosen_terms
    select_term_by_freq(k)
    select_term_by_freq_given_time(nodes, k)
    select_term_by_degree(k)

if __name__ == "__main__":
    run("machine learning", 3, 1990, 2012, 1000, 0.1)
    run("deep learning", 3, 1990, 2012, 1000, 0.1)
    run("artificial intelligence", 3, 1990, 2012, 1000, 0.1)
    run("data mining", 3, 1990, 2012, 1000, 0.1)
    run("computer graphics", 3, 1990, 2012, 1000, 0.1)
    run("support vector machin", 3, 1990, 2012, 1000, 0.1)