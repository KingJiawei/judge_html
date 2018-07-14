import os,sys
import collections
import time
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pprint

class ExtractByTfidf:
    def __init__(self):
        self.no_html_feature_file = "/home/jiawei/judge_html/test_tfidf/no_html_feature_file.txt"
        self.html_feature_file = "/home/jiawei/judge_html/test_tfidf/html_feature_file.txt"
        self.total_feature_file = "/home/jiawei/judge_html/test_tfidf/total_feature_file.txt"
        self.no_html_split_rate = 0.7
        self.html_split_rate = 0.7
        self.all_sample_path = "/home/jiawei/judge_html/all_file"
        self.html_sample_path = "test_sample/html_file"
        self.no_html_sample_path = "test_sample/no_html_file"
        #self.html_sample_path = "/home/jiawei/judge_html/html_file"
        #self.no_html_sample_path = "/home/jiawei/judge_html/no_html_file"
        self.train_file_path = "/home/jiawei/judge_html/test_tfidf/"
        self.test_file_path = "/home/jiawei/judge_html/test_tfidf/"
        self.train_file = None
        self.test_file = None
        self.model_path = "/sa/middle_dir/judge_html"


        self.file_name = './file_name.txt'
        self.corpus = []
        self.debug_file = './all_feature.txt'

        self.yara_rule = 'tfidf.yar'

        self.tfidf_feature_dict = {}
        self.no_html_count = 0
        self.html_count = 0
        self.total_count = 0
        self.tfidf = None
        self.nmf = None
        self.lda = None
        self.vectorizer = None
        self.n_components = 2
        self.n_top_words = 100
        self.feature_list = []
        self.stop_word_list = [str(hex(i)).split('0x')[1] for i in range(0, 65536)]
        self.stop_word_hex = ['0' + i for i in self.stop_word_list]
        self.stop_word_hex2 = ['x' + i for i in self.stop_word_list]
        self.stop_word_oct = [str(oct(i)) for i in range(0, 65536)]
        self.stop_word_list.extend(self.stop_word_hex)
        self.stop_word_list.extend(self.stop_word_hex2)
        self.stop_word_list.extend(self.stop_word_oct)
        self.indices = None

    def fill_corpus(self,path):
        with open(self.file_name,'wb') as fh:
            for root, dirs, files in os.walk(path):
                for name in files:
                    file_path = os.path.join(root, name)
                    self.corpus.append(file_path)
                    fh.write(file_path)
                    fh.write("\n")

    def use_lda(self):
        self.lda = LatentDirichletAllocation(n_components=self.n_components, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        self.lda.fit(self.tfidf)
        tfidf_feature_names = self.vectorizer.get_feature_names()
        self.print_top_words(self.lda, tfidf_feature_names, self.n_top_words)

    def use_nmf(self):
        self.nmf = NMF(n_components=self.n_components, random_state=1,
                  beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
                  l1_ratio=.5).fit(self.tfidf)
        print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
        tfidf_feature_names = self.vectorizer.get_feature_names()
        self.print_top_words(self.nmf, tfidf_feature_names, self.n_top_words)

    def print_top_words(self,model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

    def record_nparray_in_file(self,html_feature_file,no_html_feature_file,np_array):
        line_num = 0
        with open(html_feature_file, 'wb') as file_open:
            for n in np_array[:self.html_count]:
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
                        print '[ERROR] cannot write , exception is {}'.format(str(e))
                line_feature = '{} {} # {}\n'.format(label, feature_msg, comments)
                file_open.write(line_feature)
                line_num += 1
        with open(no_html_feature_file, 'wb') as file_open:
            for n in np_array[self.html_count:]:
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
                        print '[ERROR] cannot write , exception is {}'.format(str(e))
                line_feature = '{} {}\n'.format(label, feature_msg)
                file_open.write(line_feature)
                line_num += 1
        os.system("cat {} > {}".format(self.html_feature_file,self.total_feature_file))
        os.system("cat {} >> {}".format(self.no_html_feature_file,self.total_feature_file))


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
        cmd = "cp {}/*.txt {}".format(self.model_path,self.train_file_path)
        os.system(cmd)


    def train_and_score(self):
        working_dir = r"/sa/githubee/md_auto_tools/src/machine_learning/training_process"
        print '[Change Working Dir] ' + working_dir
        os.chdir(working_dir)

        threshold = float (self.html_count / self.total_count)

        cmd = 'python train_html.py {}'.format(self.train_file)
        print cmd
        os.system(cmd)
        cmd = 'python predict_html.py {} {}'.format(self.train_file, threshold)
        print cmd
        os.system(cmd)
        cmd = 'python predict_html.py {} {}'.format(self.test_file, threshold)
        print cmd
        os.system(cmd)

    def select(self,train_file, result_dir):

        # Build a classification task using 3 informative features
        X, y = load_svmlight_file(train_file)

        # Build a forest and compute the feature importances
        forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

        forest.fit(X, y)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        self.indices = np.argsort(importances)[::-1]
        print self.indices
        with open(os.path.join(result_dir, 'TFIDF_Contributeword.txt'), "w+") as fs:
            for f in range(X.shape[1]):
                select_word = "{0}.feature {1} {2} ({3})".format(str(f + 1), str(self.indices[f]), str(self.feature_list[self.indices[f]]),str(importances[self.indices[f]]))
                fs.write(select_word)
                fs.write('\n')
        with open(self.yara_rule,'w') as yara_fh:
            num = 0
            for f in range(X.shape[1]):
                rule = '''
rule tfidf_keyword_{1}
{{
    meta:
        index = {1}
        tfidf = {2}
    strings:
        $s="{0}" fullword nocase
    condition:
        $s
}}'''.format(str(self.feature_list[self.indices[f]]),str(num), str(self.vectorizer.vocabulary_[str(self.feature_list[self.indices[f]])]))
                yara_fh.write(rule)
                yara_fh.write('\n')
                num += 1
                if num == 100:
                    break

    def convert_to_libsvm_format(self, label, features):
        feature_msg = ''
        if isinstance(features, dict):
            ordered_features = collections.OrderedDict(sorted(features.items()))
            for i in ordered_features:
                value = ordered_features[i]
                if float(value) > 0:
                    feature_msg += '{}:{} '.format(i, value)
        else:
            feature_msg = features
        return '{} {}\n'.format(label, feature_msg)

    def start_extract(self):
        self.fill_corpus(self.html_sample_path)
        self.html_count = len(self.corpus)
        self.fill_corpus(self.no_html_sample_path)
        self.total_count = len(self.corpus)

        self.vectorizer = TfidfVectorizer(input='filename',decode_error='ignore',max_features=300,ngram_range=(1,3),stop_words=self.stop_word_list)
        self.tfidf = self.vectorizer.fit_transform(self.corpus)
        tfidf_array = self.tfidf.toarray()

        self.record_nparray_in_file(self.html_feature_file,self.no_html_feature_file, tfidf_array)
        pprint.pprint(self.vectorizer.vocabulary_)
        with open(self.debug_file, 'wb') as log_open:
            vocabulary_dict = collections.OrderedDict(sorted(self.vectorizer.vocabulary_.items(),key=lambda x:x[1]))
            for key in vocabulary_dict.keys():
                try:
                    log_open.write(key)
                    log_open.write("\n")
                    self.feature_list.append(key)
                except Exception,e:
                    print '[ERROR] exception is {}'.format(str(e))
        self.select(self.total_feature_file,'.')
        tfidf_array_copy = tfidf_array.copy()
        np.savetxt("array.txt", tfidf_array_copy)
        indice = np.array(self.indices[:100])
        tfidf_array_select = tfidf_array_copy[:, indice]
        tfidf_array_select[tfidf_array_select > 0] = 1
        tfidf_array_int = tfidf_array_select.astype(dtype=np.int8)
        (x,y) = tfidf_array_int.shape
        print x,y
        tfidf_base_index = 10
        with open("tfidf_select.txt", 'wb') as output:
            for i in range(0, x):
                line_dict = {}
                for j in range(0, y):
                    line_dict[tfidf_base_index + j] = tfidf_array_int[i, j]
                feature_line = self.convert_to_libsvm_format(1, line_dict)
                output.write(feature_line)

        np.savetxt("array3.txt", tfidf_array_int)

        self.split_file()

        self.build_train_test_set()

        self.train_and_score()
        self.save_model()


        """
        print self.vectorizer.vocabulary_
        print tfidf_array.shape
        print "##################NMF#####################"
        self.use_nmf()
        print "##################LDA#####################"
        self.use_lda()
        """

if __name__ == "__main__":
    extract_by_tfidf = ExtractByTfidf()
    extract_by_tfidf.start_extract()
