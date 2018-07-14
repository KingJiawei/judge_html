import gensim
from gensim import corpora,models,similarities
import os,sys
import codecs
import pprint

class MyCorpus(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for name in files:
                file_path = os.path.join(root, name)
                file_open = codecs.open(file_path, 'rb',encoding='utf8', errors='ignore').read().split()
                yield file_open

class ExtractByLda:
    def __init__(self):
        self.corpura = None
        self.html_path = './html_file'
        self.no_html_path = './no_html_file'

    def fill_corpora(self):
        self.corpura = MyCorpus(self.html_path)

    def start(self):
        self.fill_corpora()
        dictionary = corpora.Dictionary(self.corpura)
        #print dictionary
        new_doc = "html body div"
        new_vec = dictionary.doc2bow(new_doc.lower().split())
        print(new_vec)
        print(dictionary.token2id)
        #tfidf = models.TfidfModel()




if __name__ == "__main__":
    extract_by_lda = ExtractByLda()
    extract_by_lda.start()