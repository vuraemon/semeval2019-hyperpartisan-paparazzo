#!/usr/bin/env python
# coding: utf-8


import getopt
import lxml.sax
import lxml.etree
import numpy as np
import os
import re
import sys
import xml.sax
import pickle as pkl
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model
import scipy.sparse as sp
import string
import networkx as nx
import spacy
import itertools
import bs4
import sklearn.feature_extraction.text as text
from multiprocessing import Pool
from itertools import chain
nlp=spacy.load('en_core_web_lg')


def n_grams(tokens, n=1):
    shiftToken = lambda i: (el for j,el in enumerate(tokens) if j>=i)
    shiftedTokens = (shiftToken(i) for i in range(n))
    tupleNGrams = zip(*shiftedTokens)
    return (" ".join(i) for i in tupleNGrams)


def range_ngrams(tokens, ngramRange=(1,4)):
    return chain(*(n_grams(tokens, i) for i in range(*ngramRange)))


def tree_to_string(edge_dfs, root_label):
    vocabulary,childs,parents,id_word ={},{},{},{}
    
    for edge in list(edge_dfs):
        if edge[0] not in vocabulary:
            vocabulary[edge[0]]=len(vocabulary)
            id_word[vocabulary[edge[0]]]=edge[0].split(' ')[0]

        if edge[1] not in vocabulary:
            vocabulary[edge[1]]=len(vocabulary)
            id_word[vocabulary[edge[1]]]=edge[1].split(' ')[0]

        parents[vocabulary[edge[1]]]=vocabulary[edge[0]]
        if vocabulary[edge[0]] not in childs: childs[vocabulary[edge[0]]]=[vocabulary[edge[1]]]
        else: childs[vocabulary[edge[0]]].append(vocabulary[edge[1]])

    root=vocabulary[root_label]
    parents[root]=-1
    if -1 not in childs: childs[-1]=[root]
    else: childs[-1].append(root)
        
    forwards=[]
    vist=np.ones(len(vocabulary))
    while True:
        if np.sum(vist)==0: break
        for node in range(len(vocabulary)):
            if vist[node]==1:
                if node not in childs:
                    forwards.append(node)
                    vist[node]=0
                elif np.sum([vist[child] for child in childs[node]])==0:
                    forwards.append(node)
                    vist[node]=0
                    
    dept=np.zeros(len(vocabulary),dtype=int)
    dept_node={0: [root]}
    for node in forwards[::-1][1:]:
        dept[node]=dept[parents[node]] + 1
        if dept[node] not in dept_node: dept_node[dept[node]]=[node]
        else: dept_node[dept[node]].append(node)
            
    root=forwards[-1]
    result=[{} for i in forwards]
    max_dept=np.max(list(dept_node.keys()))
    for dept in range(max_dept, -1, -1):
        votmp={noe:id_word[noe] for noe in dept_node[dept]}
        for node, v in sorted(votmp.items(), key=lambda kv: kv[1]):
            result[node]['label']=id_word[node]

            if node in childs:
                votc={noe:id_word[noe] for noe in childs[node]}
                for child, v in sorted(votc.items(),key=lambda kv: kv[1]):
                    result[node]['label']+=result[child]['label']

            result[node]['label']='(' + result[node]['label']+')'
            
    return result[root]['label']


def dependency_extraction(sent):
    doc=nlp(sent)
    token_lemma = []
    token_pos = []
    characters = [tok for tok in sent.lower()]
    
    edges=[]
    for token in doc:
        token_lemma.append(token.lemma_)
        token_pos.append(token.pos_)
        for child in token.children:
            edges.append(('{0} {1}'.format(token.lemma_, token.i),'{0} {1}'.format(child.lemma_, child.i)))

    graph=nx.DiGraph(edges)
    final,roots=[],[]
    for num in range(2, 5):
        for sub_nodes in itertools.combinations(graph.nodes(), num):
            subg = graph.subgraph(sub_nodes)
            if nx.is_tree(subg):
                d = dict(subg.in_degree(subg.nodes()))
                tmp_root,two_parents=[],False

                for i,j in d.items():
                    if j==0: tmp_root.append(i)
                    if j>1: two_parents == True

                if not two_parents and len(tmp_root)==1:
                    final.append(subg)
                    roots.append(tmp_root[0])
                    
    features=[]
    
    for i in range(len(roots)): features.append('| '+tree_to_string(list(nx.edge_dfs(final[i])), roots[i]))
    for fea in range_ngrams(token_lemma): features.append('|| '+fea)
    for fea in range_ngrams(token_pos): features.append('|+ '+fea)
    for fea in range_ngrams(characters): features.append('|@ '+fea)
    for fea in doc.ents: features.append('|$ '+str(fea.label_)+' '+str(fea))
    
    return features


def is_pure_p_tag(content):
    childrens=content.findChildren()
    if len(childrens)==0:
        return 'pure_p',None
    else:
        return 'not_pure_p',childrens


def classify_content(content):
    if type(content) is bs4.element.Tag:
        return 'Tag'
    elif type(content) is bs4.element.NavigableString:
        if content.replace(' ', '')is'':
            return 'SpaceString'        
        return 'NavigableString'


def trim_string(s):
    r=s.strip()
    r=re.sub(' +',' ',r)
    return r


def extract_tag(contents):
    string_pos=''
    text_pos=[]
    for content in contents:
        if classify_content(content)is'NavigableString':
            string_pos+=content+' '
        elif classify_content(content)is'Tag':
            string_pos+=content.text+' '
            
    string_pos=string_pos.replace('&amp;', ' & ')
    return trim_string(string_pos)


class CustomCountVectorizer(text.TfidfVectorizer):
    def __init__(self):
        super(CustomCountVectorizer, self).__init__()
        self.min_df=0.05
        self.max_df=0.95

        
    def build_analyzer(self):
        return lambda doc: self.features_extraction(doc)
    
    
    def combined_extraction(self,sents):
        features = []
        
        features_list = Pool(processes=4).map(dependency_extraction, sents)
        for feas in features_list: features.extend(feas)
            
        return features
    
    
    def features_extraction(self,doc):
        soup=bs4.BeautifulSoup(doc,'xml')
        features,paragraphs,paragraphs_nontag=[],[],[]
        title=soup.article.attrs['title']
        print(soup.article.attrs['id'])
        
        for content in soup.article.contents:
            if classify_content(content) is 'Tag':
                str_pos=extract_tag(content)
                if str_pos.replace(' ','')=='': continue
                if str_pos not in paragraphs: paragraphs.append(str_pos)
            else:
                ct = trim_string(content)
                if ct not in paragraphs_nontag: paragraphs_nontag.append(ct)
                
        sents = []
        for paragraph in paragraphs:
            tokens = nlp(paragraph)
            for sent in tokens.sents: sents.append(str(sent))
                
        features.extend(self.combined_extraction(sents))
            
        sents = []
        for paragraph in paragraphs_nontag:
            tokens = nlp(paragraph)
            for sent in tokens.sents: sents.append(str(sent))
                
        for fea in self.combined_extraction(sents): features.append('@_@'+fea)
            
        title_feas=self.combined_extraction([title])
        features.extend(title_feas)
        
        for fea in title_feas: features.append('^_^'+fea)
            
        return features



def parse_options():
    try:
        long_options = ["inputDataset=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "i:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-i", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -I or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, outputDir)


def read_the_dataset(inputRunFile):
    articleId=[]
    articles=[]
    articleIdDict={}
    
    class HyperpartisanNews(xml.sax.ContentHandler):
        def __init__(self):
            xml.sax.ContentHandler.__init__(self)
            self.lxmlhandler='undefined'

        def startElement(self,name,attrs):
            if name!='articles':
                if name=='article':
                    self.lxmlhandler=lxml.sax.ElementTreeContentHandler()
                self.lxmlhandler.startElement(name,attrs)

        def characters(self,data):
            if self.lxmlhandler!='undefined':
                self.lxmlhandler.characters(data)
              
        def handleArticle(self,article):   
            text=lxml.etree.tostring(article,method='xml',encoding='unicode')
            aid=article.get('id')
            articleIdDict[aid]=len(articleId)
            articles.append(text)
            articleId.append(aid)

        def endElement(self,name):
            if self.lxmlhandler!='undefined':
                self.lxmlhandler.endElement(name)
                if name=='article':
                    self.handleArticle(self.lxmlhandler.etree.getroot())
                    self.lxmlhandler='undefined'
                  
    xml.sax.parse(inputRunFile,HyperpartisanNews())
    
    return articleId,articles


def main(inputDataset, outputDir):
    runOutputFileName='prediction.txt'
    for file in os.listdir(inputDataset):
        if file.endswith('.xml'):
            with open(inputDataset + '/' + file) as inputRunFile:
                ID,X=read_the_dataset(inputRunFile)
         
    f=open('final_predictor.pkl', 'rb')
    predictor=pkl.load(f)
    f.close()
    
    vectorizer=predictor['vectorizer']
    model=predictor['model']
    
    with open(outputDir + "/" + runOutputFileName, 'a') as outFile:
        for idx,val in enumerate(ID):
            X_vec=vectorizer.transform([X[idx]])
            proba=model.predict(X_vec)

            lbl='false'
            if proba[0]==1:
                lbl='true'

            print(val + ' ' + lbl + ' ' + '1.000000000000000', file=outFile)

if __name__ == '__main__':
    main(*parse_options())