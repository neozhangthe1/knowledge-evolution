#!env python2

import rpc_pb2
import interface_pb2
import zmq

context = zmq.Context()


def request(server, method, params):
    socket = context.socket(zmq.REQ)
    socket.connect(server)

    request = rpc_pb2.Request()
    request.method = method
    if type(params) is not str:
        request.param = params.SerializeToString()
    else:
        request.param = params

    socket.send(request.SerializeToString())
    reply = socket.recv()

    response = rpc_pb2.Response.FromString(reply)
    return response


class DataCenterClient(object):
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def searchAuthors(self, query, returned_fields=["naid", "names", "email", "pub_count", "h_index"]):
        params = interface_pb2.StringQueryParams()
        params.query = query
        params.offset = 0
        params.count = 50
        params.returned_fields.extend(returned_fields)
        method = "AuthorService_searchAuthors"
        response = request(self.endpoint, method, params)
        result = interface_pb2.AuthorResult.FromString(response.data)
        return result

    def searchPublications(self, query, returned_fields=["id", "title", "year", "conf_id", 
                                                         "jconf_name", "abs", "n_citations", 
                                                         "author_ids", "authors", "topic",
                                                         "cite_pubs", "cited_by_pubs"]):
        params = interface_pb2.StringQueryParams()
        params.query = query
        params.offset = 0
        params.count = 1000
        params.returned_fields.extend(returned_fields)
        method = "PubService_searchPublications"
        response = request(self.endpoint, method, params)
        result = interface_pb2.PublicationResult.FromString(response.data)
        return result

    def getPublicationsById(self, ids, returned_fields=["id", "title", "year", "conf_id", 
                                                         "jconf_name", "abs", "n_citations", 
                                                         "author_ids", "authors", "topic",
                                                         "cite_pubs", "cited_by_pubs"]):
        params = interface_pb2.IntQueryParams()
        params.ids.extend(ids)
        params.offset = 0
        params.count = 100000
        params.returned_fields.extend(returned_fields)
        method = "PubService_getPublicationsById"
        response = request(self.endpoint, method, params)
        result = interface_pb2.PublicationResult.FromString(response.data)
        return result

    def getPublicationsByAuthorId(self, ids, returned_fields=["id", "title", "year", "conf_id", 
                                                         "jconf_name", "abs", "n_citations", 
                                                         "author_ids", "authors", "topic",
                                                         "cite_pubs", "cited_by_pubs"]):
        params = interface_pb2.IntQueryParams()
        params.ids.extend(ids)
        params.offset = 0
        params.count = 1000
        params.returned_fields.extend(returned_fields)
        method = "PubService_getPublicationsByNaId"
        response = request(self.endpoint, method, params)
        result = interface_pb2.PublicationResult.FromString(response.data)
        return result

    def getPublicationsByConfId(self, ids, returned_fields=["id", "title", "year", "conf_id", 
                                                         "jconf_name", "abs", "n_citations", 
                                                         "author_ids", "authors", "topic",
                                                         "cite_pubs", "cited_by_pubs"]):
        params = interface_pb2.IntQueryParams()
        params.ids.extend(ids)
        params.offset = 0
        params.count = 1000000
        params.returned_fields.extend(returned_fields)
        method = "PubService_getPublicationsByConfId"
        response = request(self.endpoint, method, params)
        result = interface_pb2.PublicationResult.FromString(response.data)
        return result

    def getTopicDistributionGivenPub(self, id):
        params = interface_pb2.IntQueryParams()
        params.ids.extend(id)
        # params.returned_fields.extend(returned_fields)
        method = "ACTService_getTopicDistributionGivenPub"
        response = request(self.endpoint, method, params)
        result = interface_pb2.PublicationResult.FromString(response.data)
        return result    

    def getTopic(self, nid):
        params = str(nid)
        # params.returned_fields.extend(returned_fields)
        method = "ACTService_getTopicDistributionGivenAuthorNAid"
        response = request(self.endpoint, method, params)
        result = interface_pb2.Distribution.FromString(response.data)
        return result 

    def getTopicDistributionGivenQuery(self, query):
        params = interface_pb2.StringQueryParams()
        params.query = query
        params.offset = 0
        params.count = 50
        params.returned_fields.extend(returned_fields)
        method = "AuthorService_searchAuthors"
        response = request(self.endpoint, method, params)
        result = interface_pb2.Distribution.FromString(response.data)
        return result       


def extractPublication(p):
    au = "0"
    if len(p.author_ids) > 0:
        au = p.author_ids[0]
    dt = datetime.datetime(p.year, 1, 1, 1, 1)
    t = int(time.mktime(dt.timetuple()))
    key_terms = terms[str(p.id)].keys()
    kt = ""
    for k in key_terms:
        if kt != "":
            kt += ","
        k = k.lower()
        kt += k
    children = []
    children_ids = []
    parents = []
    parents_ids = []
    for x in p.cited_by_pubs:
        children.append(str(x))
        children_ids.append(x)
    for x in p.cite_pubs:
        parents.append(str(x))
        parents_ids.append(x)
    y = [str(p.id), str(p.id), str(au), children, 0, 
            t, t, p.n_citations, 
            p.n_citations, p.n_citations, p.authors, p.title, kt]
    return y, children_ids, parents_ids, key_terms

def getPapersCitation():
    from collections import defaultdict
    from dcclient import DataCenterClient
    f = open("E:\\vis.txt")
    f.next()
    ids = []
    for line in f:
        x = line.split("\t")
        ids.append(int(x[0]))
    c = DataCenterClient("tcp://10.1.1.211:32011")
    x = c.getPublicationsById(ids)
    id_set = set(ids)
    count = 0
    citation = defaultdict(set)
    for p in x.publications:
        for y in p.cite_pubs:
            if y in id_set:
                print count
                count += 1
                citation[p.id].add(y)
        for y in p.cited_by_pubs:
            if y in id_set:
                print count
                count += 1
                citation[y].add(p.id) 
    f_out = open("E:\\citation.txt","w")
    for p in citation:
        for q in citation[p]:
            f_out.write("%s\t%s\n"%(p,q))



def getPapersAbstract():
    from collections import defaultdict
    from dcclient import DataCenterClient
    f = open("E:\\ids.txt")
    f.next()
    ids = []
    import codecs
    f_out = codecs.open("E:\\abstracts_1.txt","w",encoding="utf-8")
    from bs4 import UnicodeDammit
    for line in f:
        x = line.split("\n")
        ids.append(int(x[0]))
    print len(ids)
    c = DataCenterClient("tcp://10.1.1.211:32011")
    for i in range(len(ids)/1000):
        print "DUMP %s"%(i*1000)
        x = c.getPublicationsById(ids[i*1000:(i+1)*1000])
        id_set = set(ids)
        count = 0
        abs = {}
        conf = {}
        authors = {}
        title = {}
        year = {}
        for p in x.publications:
            abs[p.id] = p.abs.replace("\n"," ").replace("\t"," ")
            conf[p.id] = p.jconf_name
            authors[p.id] = ",".join([str(a) for a in p.author_ids])
            title[p.id] = p.title
            year[p.id] = p.year
        for p in abs:
            if len(abs[p]) > 2:
                f_out.write("%s\n%s\n%s\n%s\n%s\n%s\n"%(p,year[p],conf[p],authors[p],title[p],UnicodeDammit(abs[p]).markup))


def getPapersAbstractYearConf():
    from collections import defaultdict
    from dcclient import DataCenterClient
    import codecs
    from bs4 import UnicodeDammit
    import os
    f = open("E:\\ids.txt")
    f.next()
    ids = []
    for line in f:
        x = line.split("\n")
        ids.append(int(x[0]))
    c = DataCenterClient("tcp://10.1.1.211:32011")
    def createFile(year, conf):
        if not os.path.exists(str(year)):
            os.makedirs(str(year))
        return codecs.open(os.path.join(str(year),conf),"w",encoding="utf-8")    
    #files = defaultdict(dict)
    files = {}
    for i in range(len(ids)/10000):
        print "DUMP %s"%(i*10000)
        x = c.getPublicationsById(ids[i*10000:(i+1)*10000])
        id_set = set(ids)
        count = 0
        abs = {}
        conf = {}
        title = {}
        year = {}
        for p in x.publications:
            abs[p.id] = p.abs.replace("\n"," ").replace("\t"," ")
            conf[p.id] = p.jconf_name#.replace("/"," ").replace("*"," ")
            title[p.id] = p.title
            year[p.id] = p.year
        for p in abs:
            if len(abs[p]) > 2 and len(conf[p]) > 1:
                #if not files[year[p]].has_key(conf[p]):
                if not files.has_key(year[p]):
                    files[year[p]] = codecs.open(str(year[p]),"w",encoding="utf-8")
                    #files[year[p]][conf[p]] = createFile(year[p], conf[p])
                file = files[year[p]]
                file.write("%s\n%s\n%s\n%s\n"%(p, conf[p], title[p], UnicodeDammit(abs[p]).markup))

def getPapersAbstractYearConf1():
    from collections import defaultdict
    from dcclient import DataCenterClient
    import codecs
    from bs4 import UnicodeDammit
    import os
    f = open("E:\\ids.txt")
    f.next()
    ids = []
    for line in f:
        x = line.split("\n")
        ids.append(int(x[0]))
    c = DataCenterClient("tcp://10.1.1.211:32011")
    def createFile(year, conf):
        if not os.path.exists(str(year)):
            os.makedirs(str(year))
        return codecs.open(os.path.join(str(year),conf),"w",encoding="utf-8")    
    #files = defaultdict(dict)
    files = {}
    confs = [5056,4906,4276,1935,2651,3399,3183,4650,2039,1938,710,3489]
    for y in confs:
        x = c.getPublicationsByConfId([y])
        abs = {}
        conf = {}
        title = {}
        year = {}
        for p in x.publications:
            abs[p.id] = p.abs.replace("\n"," ").replace("\t"," ")
            conf[p.id] = p.jconf_name#.replace("/"," ").replace("*"," ")
            title[p.id] = p.title
            year[p.id] = p.year
        for p in abs:
            if len(abs[p]) > 2 and len(conf[p]) > 1:
                #if not files[year[p]].has_key(conf[p]):
                if not files.has_key(year[p]):
                    files[year[p]] = codecs.open("pubs\\"+str(year[p]),"w",encoding="utf-8")
                    #files[year[p]][conf[p]] = createFile(year[p], conf[p])
                file = files[year[p]]
                file.write("%s\n%s\n%s\n%s\n"%(p, conf[p], title[p], UnicodeDammit(abs[p]).markup))
    for f in files:
        files[f].close()
            
                

def getCitationNetwork():
    import time
    import datetime
    from collections import defaultdict
    c = DataCenterClient("tcp://10.1.1.211:32011")
    x =  c.searchPublications("deep learning")
    data_fields = ["id", "mid", "uid", 
                   "parent", "type", "t", 
                   "user_created_at", "followers_count", "statuses_count", 
                   "friends_count", "username", "text", "words", "verified", "emotion"];
    items = []
    cite_pubs = []
    key_terms = defaultdict(int)
    year_terms = defaultdict(lambda: defaultdict(int))
    for p in x.publications:
        if p.year <= 1970:
            continue
        item, children, parents, kt = extractPublication(p)
        if len(children) > 0:
            items.append(item)
        cite_pubs.extend(children)
        cite_pubs.extend(parents)
        for k in kt:
            key_terms[k.lower()] += 1
            year_terms[p.year][k.lower()] += 1
    cite_pubs = list(set(cite_pubs))
    x = c.getPublicationsById(cite_pubs)
    for p in x.publications:
        if p.year <= 1970:
            continue
        item, children, parents, kt = extractPublication(p)
        if len(children) > 0 and len(children) > 0:
            items.append(item)
        cite_pubs.extend(children)
        for k in kt:
            key_terms[k.lower()] += 1
            year_terms[p.year][k.lower()] += 1

    sorted_key_terms = sorted(key_terms.items(), key = lambda x: x[1], reverse = True)
        
    import json
    dump = open("pubs_dump.json","w")
    d = json.dumps(items)
    dump.write(d)
    dump.close()

def getTopicTrend():
    import time
    import datetime
    from collections import defaultdict
    c = DataCenterClient("tcp://10.1.1.211:32011")
    x =  c.searchAuthors("deep learning")
    data_fields = ["id", "mid", "uid", 
                   "parent", "type", "t", 
                   "user_created_at", "followers_count", "statuses_count", 
                   "friends_count", "username", "text", "words", "verified", "emotion"];
    pubs = []
    key_terms = defaultdict(int)
    year_terms = defaultdict(lambda: defaultdict(int))
    for a in x.authors:
        result = c.getPublicationsByAuthorId([a.naid])
        for p in result.publications:
            if p.year > 1970:
                item, children, kt = extractPublication(p)
                pubs.append(item)
            for k in kt:
                key_terms[k.lower()] += 1
                year_terms[p.year][k.lower()] += 1

    sorted_key_terms = sorted(key_terms.items(), key = lambda x: x[1], reverse = True)
    year_tfidf = defaultdict(list)
    for y in year_terms:
        for k in year_terms[y]:
            if key_terms[k] > 1:
                year_tfidf[y].append({"text":k, "size":float(year_terms[y][k])/key_terms[k]})
        
    import json
    dump = open("pubs_dump.json","w")
    d = json.dumps(items)
    dump.write(d)
    dump.close() 

    dump = open("year_tfidf.json","w")
    d = json.dumps(year_tfidf)
    dump.write(d)
    dump.close()

def process_terms():
    term_freq = defaultdict(int)
    term_year_freq = defaultdict(lambda: defaultdict(int))
    for p in terms:
        for k in terms[p]:
            term_freq[k.lower()] += 1
    sorted_term_freq = sorted(term_freq.items(), key = lambda x: x[1], reverse = True)
            


def main():
    import time
    import datetime
    c = DataCenterClient("tcp://10.1.1.211:32011")
    x =  c.searchPublications("data mining")
    data_fields = ["id", "mid", "uid", 
                   "parent", "type", "t", 
                   "user_created_at", "followers_count", "statuses_count", 
                   "friends_count", "username", "text", "words", "verified", "emotion"];
    items = []
    for p in x.publications:
        au = "0"
        if len(p.author_ids) > 0:
            au = p.author_ids[0]
        dt = datetime.datetime(p.year, 1, 1, 1, 1)
        t = int(time.mktime(dt.timetuple()))
        children = []
        parents = []
        for x in p.cited_by_pubs:
            children.append(str(x))
        y = [str(p.id), str(p.id), str(au), children, 0, 
             t, t, p.n_citations, 
             p.n_citations, p.n_citations, p.authors, p.title, "hello,world"]
        items.append(y)
    import json
    dump = open("pubs_dump.json","w")
    d = json.dumps(items)
    dump.write(d)
    dump.close()

    import pickle
    terms = pickle.load(open("..\\static\\pickle\\terms_dump_all.pickle"))



if __name__ == "__main__":
    main()
