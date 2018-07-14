import os,sys
import re
import collections
import time
import shutil
import yara
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from encoding_checker import *


class ExtractByKeywordYara:
    def __init__(self):
        self.keyword_file = "keyword.yar"
        self.feature_list = []
        self.feature_dict = {}
        self.html_pattern_list = []

        self.no_html_feature_file = "/home/jiawei/judge_html/test_keyword_by_tfidf/no_html_feature_file.txt"
        self.html_feature_file = "/home/jiawei/judge_html/test_keyword_by_tfidf/html_feature_file.txt"
        self.no_html_tfidf_feature_file = "/home/jiawei/judge_html/test_keyword_by_tfidf/no_html_tfidf_feature_file.txt"
        self.html_tfidf_feature_file = "/home/jiawei/judge_html/test_keyword_by_tfidf/html_tfidf_feature_file.txt"
        self.total_tfidf_feature_file = "/home/jiawei/judge_html/test_keyword_by_tfidf/total_tfidf_feature_file.txt"

        self.no_html_split_rate = 0.7
        self.html_split_rate = 0.7

        self.html_sample_path = "/home/jiawei/judge_html/html_file"
        self.no_html_sample_path = "/home/jiawei/judge_html/no_html_file"

        #self.html_sample_path = "test_sample/html_file"
        #self.no_html_sample_path = "test_sample/no_html_file"

        self.train_file_path = "/home/jiawei/judge_html/test_keyword_by_tfidf/"
        self.test_file_path = "/home/jiawei/judge_html/test_keyword_by_tfidf/"

        self.train_file = None
        self.test_file = None
        self.model_path = "/sa/middle_dir/judge_html"

        self.yara_rules_ = yara.compile(self.keyword_file)
        self.matched_feature_index_map_ = {}
        self.total_num = 0
        self.html_num = 0
        self.no_html_num = 0

        self.corpus = []
        self.stop_word_list = [str(hex(i)).split('0x')[1] for i in range(0, 65536)]
        self.stop_word_hex = ['0' + i for i in self.stop_word_list]
        self.stop_word_hex2 = ['x' + i for i in self.stop_word_list]
        self.stop_word_oct = [str(oct(i)) for i in range(0, 65536)]
        self.stop_word_list.extend(self.stop_word_hex)
        self.stop_word_list.extend(self.stop_word_hex2)
        self.stop_word_list.extend(self.stop_word_oct)

        self.n_top_words = 100
        self.tfidf_base_index = 820
        self.file_index = 0
        self.tfidf_feature_list = []
        self.corpus_content = []

        self.encoding_modifier_ = EncodingModifier()

    def generate_feature_list(self):
        file_open = open(self.keyword_file,'rb')
        for line in file_open:
            if line is not None and line not in self.feature_list:
                self.feature_list.append(line.strip())

    def analyze_content(self, content):
        self.matched_feature_index_map_ = {}
        matched_rules = self.yara_rules_.match(data=content)
        for matched in matched_rules:
            index_value = matched.meta['index']
            self.matched_feature_index_map_[index_value] = 1
        return self.matched_feature_index_map_

    def get_features(self):
        return self.matched_feature_index_map_

    def convert_to_libsvm_format(self, label, features, comments):
        feature_msg = ''
        if isinstance(features, dict):
            ordered_features = collections.OrderedDict(sorted(features.items()))
            for i in ordered_features:
                value = ordered_features[i]
                if float(value) > 0:
                    feature_msg += '{}:{} '.format(i, value)
        else:
            feature_msg = features
        return '{} {} # {}\n'.format(label, feature_msg, comments)

    def build_train_test_set(self):
        parent_dir, filename = os.path.split(self.no_html_feature_file)
        filename_wo_ext, ext = os.path.splitext(filename)
        no_html_part_a = os.path.join(parent_dir, filename_wo_ext + '_random_{}_group_a{}'.format(self.no_html_split_rate, ext))
        no_html_part_b = os.path.join(parent_dir, filename_wo_ext + '_random_{}_group_b{}'.format(self.no_html_split_rate, ext))

        parent_dir, filename = os.path.split(self.html_feature_file)
        filename_wo_ext, ext = os.path.splitext(filename)
        html_part_a = os.path.join(parent_dir, filename_wo_ext + '_random_{}_group_a{}'.format(self.html_split_rate, ext))
        html_part_b = os.path.join(parent_dir, filename_wo_ext + '_random_{}_group_b{}'.format(self.html_split_rate, ext))

        self.train_file = os.path.join(self.train_file_path, 'training_set.txt')
        self.test_file = os.path.join(self.test_file_path,'test_set.txt')

        os.system('cat {} > {}'.format(no_html_part_a, self.train_file))
        os.system('cat {} >> {}'.format(html_part_a, self.train_file))
        os.system('cat {} > {}'.format(no_html_part_b, self.test_file))
        os.system('cat {} >> {}'.format(html_part_b, self.test_file))

    def split_file(self):
        working_dir = r"/sa/githubee/md_auto_tools/src/machine_learning/preprocess"
        print '[Change Working Dir] ' + working_dir
        os.chdir(working_dir)

        cmd = 'python split.py {} {}'.format(self.no_html_feature_file, self.no_html_split_rate)
        print '[Split Set] ' + cmd
        os.system(cmd)
        cmd = 'python split.py {} {}'.format(self.html_feature_file, self.html_split_rate)
        print '[Split Set] ' + cmd
        os.system(cmd)

    def save_model(self):
        try:
            model_path = os.path.join(self.model_path, 'xgb.model')
            while 1:
                if os.path.exists(model_path):
                    html_size1 = os.path.getsize(model_path)
                    time.sleep(3)
                    html_size2 = os.path.getsize(model_path)
                    print "model_size1 is %d,model_size2 is %d,delta is %d", (
                    html_size1, html_size2, html_size2 - html_size1)
                    if (html_size2 == html_size1):
                        break
            print "get model from", model_path
            shutil.copy2(model_path, self.train_file_path)
        except:
            print "save model failed!"
        cmd = "cp {}/*.txt {}".format(self.model_path, self.train_file_path)
        os.system(cmd)

    def compute_threshold(self,x,y):
        return float(x)/float(x + y)

    def train_and_score(self):
        working_dir = r"/sa/githubee/md_auto_tools/src/machine_learning/training_process"
        print '[Change Working Dir] ' + working_dir
        os.chdir(working_dir)
        cmd = 'python train_html.py {}'.format(self.train_file)
        print cmd
        os.system(cmd)
        cmd = 'python predict_html.py {} {}'.format(self.train_file,self.threshold)
        print cmd
        os.system(cmd)
        cmd = 'python predict_html.py {} {}'.format(self.test_file,self.threshold)
        print cmd
        os.system(cmd)

    def record_nparray_in_file(self,html_tfidf_feature_file,no_html_tfidf_feature_file,np_array):
        line_num = 0
        with open(html_tfidf_feature_file, 'wb') as file_open:
            for n in np_array[:self.html_num]:
                feature_msg = ''
                comments = None
                label = 1
                col = 0
                for value in n:
                    try:
                        if float(value) > 0:
                            feature_msg += '{}:{} '.format(col, value)
                        col += 1
                        comments = self.corpus[line_num]
                    except Exception, e:
                        print '[ERROR] cannot write{} , exception is {}'.format(comments,str(e))
                line_feature = '{} {} # {}\n'.format(label, feature_msg, comments)
                file_open.write(line_feature)
                line_num += 1
        with open(no_html_tfidf_feature_file, 'wb') as file_open:
            for n in np_array[self.html_num:]:
                feature_msg = ''
                comments = None
                label = 0
                col = 0
                for value in n:
                    try:
                        if float(value) > 0:
                            feature_msg += '{}:{} '.format(col, value)
                        col += 1
                        comments = self.corpus[line_num]
                    except Exception, e:
                        print '[ERROR] cannot write{} , exception is {}'.format(comments, str(e))
                line_feature = '{} {}\n'.format(label, feature_msg)
                file_open.write(line_feature)
                line_num += 1
        os.system("cat {} > {}".format(self.html_tfidf_feature_file,self.total_tfidf_feature_file))
        os.system("cat {} >> {}".format(self.no_html_tfidf_feature_file,self.total_tfidf_feature_file))

    def select_feature(self, train_file, result_dir):

        # Build a classification task using 3 informative features
        X, y = load_svmlight_file(train_file)

        # Build a forest and compute the feature importances
        forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

        forest.fit(X, y)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        self.indices = np.argsort(importances)[::-1]
        with open("log.txt", 'wb') as log_open:
            vocabulary_dict = collections.OrderedDict(sorted(self.vectorizer.vocabulary_.items(),key=lambda x:x[1]))
            for key in vocabulary_dict.keys():
                try:
                    log_open.write(key)
                    log_open.write("\n")
                    self.tfidf_feature_list.append(key)
                except Exception,e:
                    print '[ERROR] exception is {}'.format(str(e))
        with open(os.path.join(result_dir, 'TFIDF_Contributeword.txt'), "w+") as fs:
            for f in range(X.shape[1]):
                select_word = "{0}.feature {1} {2} ({3})".format(str(f + 1), str(self.indices[f]),
                                                                     str(self.tfidf_feature_list[self.indices[f]]),
                                                                     str(importances[self.indices[f]]))
                fs.write(select_word)
                fs.write('\n')


    def extract_tfidf_keyword(self):
        self.vectorizer = TfidfVectorizer(input='content', decode_error='ignore', max_features=300, ngram_range=(2, 3),
                                          stop_words=self.stop_word_list,strip_accents='ascii')
        self.tfidf = self.vectorizer.fit_transform(self.corpus_content)
        tfidf_array = self.tfidf.toarray()
        self.record_nparray_in_file(self.html_tfidf_feature_file, self.no_html_tfidf_feature_file, tfidf_array)
        self.select_feature(self.total_tfidf_feature_file, '.')
        tfidf_array_copy = tfidf_array.copy()
        select_indice = np.array(self.indices[:self.n_top_words])
        self.tfidf_array_select = tfidf_array_copy[:, select_indice]
        self.tfidf_array_float = self.tfidf_array_select.copy()
        self.tfidf_array_select[self.tfidf_array_select > 0] = 1
        self.tfidf_array_int = self.tfidf_array_select.astype(dtype=np.int8)
        print self.tfidf_array_float
        np.savetxt("array_float.txt", self.tfidf_array_float)
        np.savetxt("array3.txt", self.tfidf_array_int)

    def add_tfidf_feature(self,np_array):
        pass

    def fill_corpus(self,path):
        for root, dirs, files in os.walk(path):
            for name in files:
                file_path = os.path.join(root, name)
                self.corpus.append(file_path)
                content = self.encoding_modifier_.modify_file(file_path)
                self.corpus_content.append(content)

    def dump_feature(self):
        self.file_index = 0
        with open(self.html_feature_file, 'wb') as file_open:
            for file_path in self.corpus[:self.html_num]:
                try:
                    content = self.encoding_modifier_.modify_file(file_path)
                    features = self.analyze_content(content)
                    tfidf_dict = {}
                    for j in range(0, self.n_top_words):
                        tfidf_dict[self.tfidf_base_index + self.indices[j]] = self.tfidf_array_int[self.file_index,j]
                    self.file_index += 1
                    features.update(tfidf_dict)
                    feature_line = self.convert_to_libsvm_format(1, features, file_path)
                    file_open.write(feature_line)
                    print "############ {} {}  ################".format(self.file_index, file_path)
                except Exception, e:
                    print '[ERROR] cannot extract feature on {}, exception is {}'.format(file_path, str(e))


        with open(self.no_html_feature_file, 'wb') as file_open:
            for file_path in self.corpus[self.html_num:]:
                try:
                    content = self.encoding_modifier_.modify_file(file_path)
                    features = self.analyze_content(content)
                    tfidf_dict = {}
                    for j in range(0, self.n_top_words):
                        tfidf_dict[self.tfidf_base_index + self.indices[j]] = self.tfidf_array_int[self.file_index, j]
                    self.file_index += 1
                    features.update(tfidf_dict)
                    feature_line = self.convert_to_libsvm_format(0, features, file_path)
                    file_open.write(feature_line)
                    print "############ {} {}  ################".format(self.file_index, file_path)
                except Exception, e:
                    print '[ERROR] cannot extract feature on {}, exception is {}'.format(file_path, str(e))


    def start(self):
        self.fill_corpus(self.html_sample_path)
        self.html_num = len(self.corpus)
        self.fill_corpus(self.no_html_sample_path)
        self.total_num = len(self.corpus)
        self.threshold = float(self.html_num) / float(self.total_num)
        print "threshold is {}".format(self.threshold)
        print "html num is {},no html num is {}".format(self.html_num, self.total_num - self.html_num)
        self.extract_tfidf_keyword()
        self.dump_feature()
        self.split_file()

        self.build_train_test_set()

        self.train_and_score()
        self.save_model()


if __name__ == "__main__":
    extract_by_keyword_yar = ExtractByKeywordYara()
    extract_by_keyword_yar.start()
