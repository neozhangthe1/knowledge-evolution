#!env python2
import socket
from bs4 import UnicodeDammit

def request(text):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    sock.connect(("10.1.1.110",6060))
    num = 5  
    sock.send(text.encode("utf-8")+"<end>")
    reply = sock.recv(99999)
    response = parse_reply(reply)
    return response

def parse_reply(reply):
    terms = {}
    for line in reply.split("\n"):
        x = line.split("\t")
        if len(x) < 2:
            continue
        try:
            terms[x[0]] = float(x[1])
        except Exception, e:
            print e
            print x
    return terms


class TermExtractorClient(object):
    #def __init__(self):
    #    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    #    self.sock.connect(("localhost",6060))

    #def request(selftext):
    #    self.sock.send(text+"<end>")
    #    reply = self.sock.recv(99999)
    #    response = parse_reply(reply)
    #    return response

    def extractTerms(self, text):
        return request(text)

    def extractTermsFromTitleAbs(self, title, abstract):
        text = title +"." +"\n" + abstract
        terms = request(text)
        return terms

    def extractBatch1(self, titles, abs):
        term_list = {}
        for id in titles:
            text = titles[id] +"." +"\n"
            if abs.has_key(id):
                text += abs[id]
            terms = request(text)
            term_list[id] = terms
        return term_list

    #def extractBatch(self, titles, abs):
    #    term_list = {}
    #    for t in titles:
    #        text = t[1] +"." +"\n"
    #        if abs.has_key(t[0]):
    #            text += abs[t[0]]
    #        terms = request(text)
    #        term_list[] = terms
    #    return term_list

    def dumpTerms(self, term_list):
        dump = open("terms.txt")
        for id in term_list:
            term_str = ""
            for term in term_list[id]:
                term_str+=term+":"+term_list[id][term]+"\t"
            dump.write("%s*%s"%(id,term_str))
        dump.close()




def main():
    c = DataCenterClient("tcp://localhost:6060")
    print c.searchAuthors("data mining")


if __name__ == "__main__":
    main()
