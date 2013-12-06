from dcclient.dcclient import DataCenterClient
from teclient.teclient import TermExtractorClient
from utils.algorithms import *
from collections import defaultdict
from bs4 import UnicodeDammit
import numpy as np
import json     
import gensim
import pickle
import networkx as nx
import logging
from sklearn.cluster import Ward, KMeans, MiniBatchKMeans, spectral_clustering
data_center = DataCenterClient("tcp://10.1.1.111:32012")
term_extractor = TermExtractorClient()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class TopicTrend(object):
    def __init__(self):
        self.stop_words = ["data set", "training data", "experimental result", 
                           "difficult learning problem", "user query", "case study", 
                           "web page", "data source", "proposed algorithm", 
                           "proposed method", "real data", "international conference",
                           "proposed approach"]

    def init_topic_trend(self):
        print "INIT TOPIC TREND"
        self.author_result = None
        #term info
        self.num_terms = 0
        self.term_list = None
        self.term_index = None
        self.term_freq = None
        self.term_freq_given_document = None
        self.term_freq_given_time = None
        self.term_freq_given_person = None
        self.term_freq_given_person_time = None
        self.co_word_maxtrix = None
        self.reverse_term_dict = None
        #author info
        self.num_authors = 0
        self.author_list = None
        self.author_index = None
        #document info
        self.num_documents = 0
        self.document_list = None
        self.document_index = None
        self.document_list_given_time = None        
        #time info
        self.time_window = None
        self.time_slides = None
        self.num_time_slides = None
        self.start_time = None
        self.end_time = None
        #cluster info
        self.num_local_clusters = 3
        self.num_global_clusters = 3
        self.local_clusters = None
        self.local_cluster_labels = None
        self.global_clusters = None
        self.global_cluster_labels = None
        self.gloabl_feature_vectors_index = None
        self.term_first_given_person = None
        self.graph = None

    """
    current method using term extractor and 2 level clustering
    """
    def query_terms(self, q, time_window=None, start_time=None, end_time=None):
        self.init_topic_trend()
        #query documents and caculate term frequence
        self.author_list = []
        self.author_index = {}
        self.num_documents = 0
        self.document_list = []
        self.document_list_given_time = defaultdict(list)
        self.document_index = {}
        self.num_documents = 0
        self.term_index = {}
        self.num_terms = 0
        print q, time_window, start_time, end_time
        if q == "big data":
            q = [q, "large scale data mining"]
        else:
            q = [q]
        self.search_author(q, time_window=time_window, start_time=start_time, end_time=end_time)
        #local clustering
        self.local_clusters = [None for i in range(self.num_time_slides)]
        self.local_cluster_labels = [None for i in range(self.num_time_slides)]
        for time in range(self.num_time_slides):
            self.local_clustering(time)
        #global clustering
        self.global_clustering_by_spectral()
        graph = self.build_graph()
        return graph

    """
    old method using topic modeling
    """
    def query_topic_trends(self, query, threshold=0.0001):
        logging.info("MATCHING QUERY TO TOPICS", query, threshold)
        query = query.lower()
        words = []
        choose_topic = defaultdict(list)
        #check if the term is in the vocabulary
        if query in self.vocab:
            print "FOUND WORD", query, self.vocab[query]
            words.append(self.vocab[query])
        #if not, check if the words in the term exists in the vocabulary
        else:
            terms = query.split(" ")
            for t in terms:
                if t in self.vocab:
                    print "FOUND WORD", t, self.vocab[t]
                    words.append(self.vocab[t]) 
        #choose topics related to the query term
        for y in self.p_topic_given_term_y:
            for t in words:
                p_topic = self.p_topic_given_term_y[y][t]
                for i in range(len(p_topic)):
                    if p_topic[i] > threshold:
                        choose_topic[y].append(i)
        print len(choose_topic), "topics are choosed"
        return self.render_topic_graph(choose_topic)   

    def search_document_by_author(self, a, start_time=0, end_time=10000):
        logging.info("querying documents for %s from %s to %s" % (a.names, start_time, end_time))
        result = data_center.getPublicationsByAuthorId([a.naid])
        logging.info("found %s documents" % len(result.publications))
        #text for extract key terms
        text = ""
        for p in result.publications:
            #update time info
            if p.year >= start_time and p.year <= end_time:
                self.set_time(p.year)
                text += (p.title.lower() + " . " + p.abs.lower() +" . ")
                #insert document
                self.append_documents(p)
        return text

    def search_author(self, q, time_window, start_time, end_time):
        print q, time_window, start_time, end_time
        self.author_result = []
        for qu in q:
            self.author_result.extend(data_center.searchAuthors(qu).authors)
        term_set = set()
        index = 0
        for a in self.author_result:
            #insert author
            self.append_authors(a)
            #search for document
            text = self.search_document_by_author(a, start_time=start_time, end_time=end_time)
            #extract terms
            terms = term_extractor.extractTerms(text)
            for t in terms:
                if t not in self.stop_words:
                    term_set.add(t)
        self.set_terms(term_set)
        #caculate term frequence
        self.caculate_term_frequence_given_document()
        #update time slides
        self.set_time_slides(time_window)
        self.caculate_term_frequence_given_time()
        self.smooth_term_frequence_given_person_by_average()

    """
    setter
    """
    #there will be 10 time window by default
    def set_time_slides_(self, time_window):
        if time_window is not None:
            self.time_window = time_window
        else:
            self.time_window = 1 + int(np.floor((float(self.end_time - self.start_time) / 11)))
        self.num_time_slides = int(np.ceil((float(self.end_time - self.start_time) / self.time_window)))
        self.time_slides = []
        cur_time = self.start_time
        for i in range(self.num_time_slides):
            cur_slide = []
            for j in range(self.time_window):
                cur_slide.append(cur_time)
                cur_time += 1
            self.time_slides.append(cur_slide)

    #the lastest year will be a standalone time slide
    def set_time_slides(self, time_window):
        logging.info("setting time slides")
        if time_window is not None:
            self.time_window = time_window
        else:
            self.time_window = 1 + int(np.floor((float(self.end_time-1 - self.start_time) / 11)))
        self.num_time_slides = int(np.ceil((float(self.end_time-1 - self.start_time) / self.time_window))) + 1
        self.time_slides = [[] for i in range(self.num_time_slides)]
        self.time_slides[self.num_time_slides-1].append(self.end_time)
        cur_time = self.end_time-1
        for i in range(self.num_time_slides-2, -1, -1):
            for j in range(self.time_window):
                self.time_slides[i].append(cur_time)
                cur_time -= 1
                print self.time_slides
                if cur_time < self.start_time:
                    logging.info("current:%s, start:%s, end:%s"%(cur_time, self.start_time, self.end_time))
                    return

    def set_time(self, time):
        if time < self.start_time or self.start_time is None:
            self.start_time = time
        if time > self.end_time or self.end_time is None:
            self.end_time = time

    def set_terms(self, term_set):
        self.term_list = list(term_set)
        index = 0
        for t in self.term_list:
            self.term_index[t] = index
            index += 1
        self.num_terms = index

    def get_time_slide(self, year):
        for i in range(self.num_time_slides):
            if year in self.time_slides[i]:
                return i

    def append_authors(self, a):
        self.author_list.append(a)
        self.author_index[a.naid] = self.num_authors
        self.num_authors += 1

    def append_documents(self, p):
        self.document_list.append(p)
        self.document_list_given_time[p.year].append(p.id)
        self.document_index[p.id] = self.num_documents
        self.num_documents += 1

    def caculate_term_frequence_given_document(self):
        self.term_freq = np.zeros(self.num_terms)
        self.term_freq_given_document = [[] for i in range(self.num_documents)]
        self.reverse_term_dict = defaultdict(list)
        for y in self.document_list_given_time:
            year_count = 0
            for d in self.document_list_given_time[y]:
                text = (self.document_list[self.document_index[d]].title.lower()
                        + " . " 
                        + self.document_list[self.document_index[d]].abs.lower())
                for t in range(self.num_terms):
                    if self.term_list[t] in text:
                        self.term_freq[t] += 1
                        self.term_freq_given_document[self.document_index[d]].append(t)
                        self.reverse_term_dict[t].append(self.document_index[d])
                        year_count += 1
            if year_count > 0:
                self.set_time(y)

    def caculate_term_frequence_given_time(self):
        self.term_freq_given_time = np.zeros((self.num_time_slides, self.num_terms))
        self.term_freq_given_person = np.zeros((self.num_terms, self.num_authors))
        self.term_freq_given_person_time = [np.zeros((self.num_terms, self.num_authors)) for i in range(self.num_time_slides)]
        self.term_first_given_person = [{} for i in range(self.num_terms)]
        for i in range(self.num_time_slides):
            for y in self.time_slides[i]:
                for d in self.document_list_given_time[y]:
                    for t in self.term_freq_given_document[self.document_index[d]]:
                        self.term_freq_given_time[i, t] += 1
                        for a in self.document_list[self.document_index[d]].author_ids:
                            if self.author_index.has_key(a):
                                self.term_freq_given_person[t, self.author_index[a]] += 1
                                self.term_freq_given_person_time[i][t, self.author_index[a]] += 1
                                if self.term_first_given_person[t].has_key(a):
                                    if self.term_first_given_person[t][a] > y:
                                            self.term_first_given_person[t][a] = y
                                else:
                                    self.term_first_given_person[t][a] = y

    def caculate_term_frequence_(self):
        #init term frequence
        self.term_freq = np.zeros(self.num_terms)
        self.term_freq_given_time = np.zeros((self.num_time_slides, self.num_terms))
        self.term_freq_given_person = np.zeros((self.num_terms, self.num_authors))
        self.term_freq_given_person_time = [np.zeros((self.num_terms, self.num_authors)) for i in range(self.num_time_slides)]
        self.term_first_given_person = [{} for i in range(self.num_time_slides)]
        for i in range(self.num_time_slides):
            for y in self.time_slides[i]:
                for d in self.document_list_given_time[y]:
                    text = (self.document_list[self.document_index[d]].title.lower() + " . " + self.document_list[self.document_index[d]].abs.lower())
                    for t in range(self.num_terms):
                        if self.term_list[t] in text:
                            self.term_freq[t] += 1
                            self.term_freq_given_time[i, t] += 1
                            for a in self.document_list[self.document_index[d]].author_ids:
                                if self.author_index.has_key(a):
                                    #logging.info("i:%s,y:%s,d:%s,text:%s,t:%s,a:%s"%(i,y,d,text,t,a))
                                    self.term_freq_given_person[t, self.author_index[a]] += 1
                                    self.term_freq_given_person_time[i][t, self.author_index[a]] += 1
                                    if self.term_first_given_person[t].has_key(a):
                                        if self.term_first_given_person[t][a] > y:
                                            self.term_first_given_person[t][a] = y
                                    else:
                                        self.term_first_given_person[t][a] = y
        self.smooth_term_frequence_given_person_by_average()

    def smooth_term_frequence_given_person_by_incremental(self):
        for i in range(1, self.num_time_slides):
            for t in range(self.num_terms):
                for a in range(self.num_authors):
                    self.term_freq_given_person_time[i][t, a] += self.term_freq_given_person_time[i-1][t, a]

    def smooth_term_frequence_given_person_by_average(self):
        for t in range(self.num_terms):
            for a in range(self.num_authors):
                avg = self.term_freq_given_person[t, a] / float(self.num_time_slides)
                for i in range(self.num_time_slides):
                    self.term_freq_given_person_time[i][t, a] += avg
        
    def local_clustering(self, time):
        num_clusters=self.num_local_clusters
        X = self.term_freq_given_person_time[time]
        num_item = len(X)
        logging.info("KMeans... item slides-%s", time)
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters).fit(X)
        logging.info("KMeans finished")
        self.local_clusters[time] = [[] for i in range(self.num_local_clusters)]
        for i, c in enumerate(kmeans.labels_):
            self.local_clusters[time][c].append(i)
        self.local_cluster_labels[time] = kmeans.labels_

    def build_global_feature_vectors(self):
        index = 0
        self.gloabl_feature_vectors_index = [{} for i in range(self.num_time_slides)]
        dim = self.num_authors
        X = np.zeros((self.num_time_slides*self.num_local_clusters, dim))
        for t in range(self.num_time_slides):
            for i, cluster in enumerate(self.local_clusters[t]):
                self.gloabl_feature_vectors_index[t][i] = index
                for w in cluster:
                     X[index] += self.term_freq_given_person_time[t][w]
                index += 1
        return X    

    def build_global_feature_vectors_by_jaccard(self):
        index = 0
        self.gloabl_feature_vectors_index = [{} for i in range(self.num_time_slides)]
        dim = self.num_time_slides*self.num_local_clusters
        items = []
        X = np.zeros((dim, dim))
        for t in range(self.num_time_slides):
            for i, cluster in enumerate(self.local_clusters[t]):
                items.append(cluster)
                self.gloabl_feature_vectors_index[t][i] = index
                index += 1
        for i in range(dim):
            for j in range(i, dim):
                sim = jaccard_similarity(items[i], items[j])
                X[i, j] == sim
                X[j, i] == sim
        return X   
    
    def build_global_feature_vectors_by_jaccard_with_weight(self):
        #weight of the term denotes by term frequence
        index = 0
        self.gloabl_feature_vectors_index = [{} for i in range(self.num_time_slides)]
        dim = self.num_time_slides*self.num_local_clusters
        items = []
        X = np.zeros((dim, dim))
        for t in range(self.num_time_slides):
            for i, cluster in enumerate(self.local_clusters[t]):
                items.append(cluster)
                self.gloabl_feature_vectors_index[t][i] = index
                index += 1
        for i in range(dim):
            for j in range(i, dim):
                sim = jaccard_similarity_with_weight(items[i], items[j], self.term_freq)
                X[i, j] == sim
                X[j, i] == sim
        return X               

    def global_clustering(self):
        num_clusters=self.num_global_clusters
        #clustering by authors as feature
        #build feature vectors
        X = self.build_global_feature_vectors()
        logging.info("Global KMeans... ")
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters).fit(X)
        logging.info("Global KMeans finished")
        self.global_clusters = [[[] for i in range(num_clusters)] for j in range(self.num_time_slides)]
        self.global_cluster_labels = [[None for i in range(self.num_local_clusters)] for j in range(self.num_time_slides)]
        labels = kmeans.labels_
        for time in range(self.num_time_slides):
            for i, cluster in enumerate(self.local_clusters[time]):
                l = labels[self.gloabl_feature_vectors_index[time][i]]
                self.global_clusters[time][l].append(i)
                self.global_cluster_labels[time][i] = l
                #for w in self.local_clusters[time][c]:
                #    self.global_clusters[l].append(w)

    def global_clustering_by_spectral(self):
        num_clusters = self.num_global_clusters
        X = self.build_global_feature_vectors_by_jaccard_with_weight()
        logging.info("Global spectral clustering...")
        spectral = spectral_clustering(X, n_clusters=num_clusters, eigen_solver='arpack')
        logging.info("Global spectral finished")
        self.global_clusters = [[[] for i in range(num_clusters)] for j in range(self.num_time_slides)]
        self.global_cluster_labels = [[None for i in range(self.num_local_clusters)] for j in range(self.num_time_slides)]
        labels = spectral
        for time in range(self.num_time_slides):
            for i, cluster in enumerate(self.local_clusters[time]):
                l = labels[self.gloabl_feature_vectors_index[time][i]]
                self.global_clusters[time][l].append(i)
                self.global_cluster_labels[time][i] = l

    def build_graph(self):
        self.graph = {"nodes":[], "links":[], "terms":[], "people":[], "documents":[]}
        global_clusters_index = {}
        index = 0
        for time in range(self.num_time_slides):
            cluster_weight_given_time = np.zeros(self.num_global_clusters)
            document_count = 0.
            for y in self.time_slides[time]:
                document_count += len(self.document_list_given_time[y])
            document_count /= len(self.time_slides[time])
            for i, cluster in enumerate(self.global_clusters[time]):
                for c in cluster:
                    for w in self.local_clusters[time][c]:
                        cluster_weight_given_time[i] += self.term_freq_given_time[time][w]
            cluster_weight_sum_given_time = sum(cluster_weight_given_time)
            if cluster_weight_sum_given_time == 0:
                cluster_weight_sum_given_time = 1
            for i, cluster in enumerate(self.global_clusters[time]):
                terms = []
                for c in cluster:
                    for w in self.local_clusters[time][c]:
                        terms.append(w)
                if len(terms) == 0:
                    continue
                sorted_terms = sorted(terms, key=lambda t: self.term_freq[t], reverse=True)
                sorted_terms_given_time = sorted(terms, key=lambda t: self.term_freq_given_time[time][t], reverse=True)
                self.graph["nodes"].append({"key":[{"term":self.term_list[k], "w":int(self.term_freq_given_time[time][k])} for k in sorted_terms_given_time], 
                                        "name":self.term_list[sorted_terms_given_time[0]],
                                        "pos":time, 
                                        "w":cluster_weight_given_time[i]/cluster_weight_sum_given_time*(document_count+1),
                                        "n":cluster_weight_given_time[i]/cluster_weight_sum_given_time,
                                        "cluster":i})
                global_clusters_index[str(time)+"-"+str(i)] = index
                index += 1
        #caculate similarity
        global_clusters_sim_target = defaultdict(dict)
        global_clusters_sim_source = defaultdict(dict)
        for time in range(1, self.num_time_slides):
            for i1, c1 in enumerate(self.global_clusters[time]):
                key1 = str(time)+"-"+str(i1)
                if global_clusters_index.has_key(key1):
                    terms1 = []
                    for c in c1:
                        for w in self.local_clusters[time][c]:
                            terms1.append(w)
                    for i2, c2 in enumerate(self.global_clusters[time-1]):
                        key2 = str(time-1)+"-"+str(i2)
                        if global_clusters_index.has_key(key2):
                            terms2 = []
                            for c in c2:
                                for w in self.local_clusters[time][c]:
                                    terms2.append(w)
                            sim = common_word_with_weight(terms1, terms2, self.term_freq)
                            if sim > 0:
                                global_clusters_sim_target[key1][key2] = sim
                                global_clusters_sim_source[key2][key1] = sim
        for key1 in global_clusters_sim_target:
            if global_clusters_index.has_key(key1):
                m1 = sum(global_clusters_sim_target[key1].values())
                for key2 in global_clusters_sim_target[key1]:
                    if global_clusters_index.has_key(key2):
                        m2 = sum(global_clusters_sim_source[key2].values())
                        self.graph["links"].append({"source":int(global_clusters_index[key2]),
                                    "target":int(global_clusters_index[key1]),
                                    "w1":global_clusters_sim_target[key1][key2]/float(m1),
                                    "w2":global_clusters_sim_target[key1][key2]/float(m2)})
        #term frequence
        sorted_terms = sorted(self.term_list, key=lambda t: self.term_freq[self.term_index[t]], reverse=True)
        for t in sorted_terms:
            term_index = self.term_index[t]
            term_year = defaultdict(list)
            for d in self.reverse_term_dict[term_index]:
                term_year[self.document_list[d].year].append(d)
            sorted_term_year = sorted(term_year.items(), key=lambda t:t[0])
            if len(sorted_term_year) == 0:
                continue
            ty = {}
            for i in range(self.start_time+1, self.end_time):
                ty[i] = 0.0
            for c in term_year:
                ty[c] = len(term_year[c])
            start_point = sorted_term_year[0][0]
            start_time = self.get_time_slide(start_point)
            start_cluster = self.global_cluster_labels[start_time][self.local_cluster_labels[start_time][term_index]]
            start_node = global_clusters_index[str(start_time)+"-"+str(start_cluster)]
            item = {"t":t, "idx":int(term_index), 
                    "freq":int(self.term_freq[term_index]), 
                    "dist":[0 for i in range(self.num_time_slides)], 
                    "year":[{"y":j, "d":ty[j]} for j in ty],
                    "cluster":[0 for i in range(self.num_time_slides)],
                    "node":[0 for i in range(self.num_time_slides)],
                    "doc":[int(d) for d in self.reverse_term_dict[term_index]],
                    "first":[{"p":p, "y":self.term_first_given_person[term_index][p]} for p in self.term_first_given_person[term_index]],
                    "start":{"year":int(start_point), 
                             "time":int(start_time), 
                             "cluster":int(start_cluster),
                             "node":int(start_node)}}
            for time in range(self.num_time_slides):
                item["dist"][time] = int(self.term_freq_given_time[time][term_index])
                local_c = self.local_cluster_labels[time][term_index]
                item["cluster"][time] = int(self.global_cluster_labels[time][local_c])
                item["node"][time] = int(global_clusters_index[str(time)+"-"+str(item["cluster"][time])])
            self.graph["terms"].append(item)
        #people
        for author in self.author_result:
            self.graph["people"].append({"id": author.naid, 
                                         "name": author.names[0], 
                                         "hindex": author.h_index,
                                         "pub_count": author.pub_count,
                                         "cite": author.citation_no})
        #document
        for i, doc in enumerate(self.document_list):
            self.graph["documents"].append({"idx":i, "id":int(doc.id), "title":doc.title, 
                                           "year":int(doc.year), "jconf":doc.jconf_name, #"abs":doc.abs,
                                           "cite":int(doc.n_citations)})#, "authors":doc.author_ids, "topic":doc.topic})
        #time slides
        self.graph["time_slides"] = self.time_slides
        return self.graph



def main():
    trend = TopicTrend()
    trend.query_terms("deep learning")


if __name__ == "__main__":
    main()