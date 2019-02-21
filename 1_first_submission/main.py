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

from sklearn.feature_extraction.text import CountVectorizer

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
            text=lxml.etree.tostring(article,method='text',encoding='unicode')
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
         
    f=open('predictor.pkl', 'rb')
    predictor=pkl.load(f)
    f.close()
    
    wordVectorizer=predictor['wordVectorizer']
    charVectorizer=predictor['charVectorizer']
    model=predictor['model']
    
    X_test_word=wordVectorizer.transform(X)
    X_test_char=charVectorizer.transform(X)
    X_test=sp.hstack((X_test_word,X_test_char),format='csr')
    proba=model.predict(X_test)
    
    with open(outputDir + "/" + runOutputFileName, 'w') as outFile:
        for idx,val in enumerate(ID):
            lbl='false'
            if proba[idx]==1:
                lbl='true'
                
            print(val + ' ' + lbl + ' ' + '1.000000000000000', file=outFile)

if __name__ == '__main__':
    main(*parse_options())