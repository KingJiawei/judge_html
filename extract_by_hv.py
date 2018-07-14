import os, sys
import collections
import time
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import HashingVectorizer


class ExtractByHv:
    def __init__(self):
        self.no_html_feature_file = "/home/jiawei/judge_html/test_hv/no_html_feature_file.txt"
        self.html_feature_file = "/home/jiawei/judge_html/test_hv/html_feature_file.txt"
        self.no_html_split_rate = 0.7
        self.html_split_rate = 0.7
        self.html_sample_path = "/home/jiawei/judge_html/old_html_file"
        self.no_html_sample_path = "/home/jiawei/judge_html/old_no_html_file"
        self.train_file_path = "/home/jiawei/judge_html/test_hv/"
        self.test_file_path = "/home/jiawei/judge_html/test_hv/"
        self.train_file = None
        self.test_file = None
        self.model_path = "/sa/middle_dir/judge_html"

        self.file_name = './file_name.txt'
        self.corpus = []
        self.debug_file = './feature_indce.txt'
        self.tfidf_feature_dict = {}
        self.no_html_count = 0
        self.html_count = 0
        self.tfidf = None
        self.nmf = None
        self.lda = None
        self.vectorizer = None
        self.n_components = 2
        self.n_top_words = 100

    def fill_corpus(self, path):
        with open(self.file_name, 'wb') as fh:
            for root, dirs, files in os.walk(path):
                for name in files:
                    file_path = os.path.join(root, name)
                    self.corpus.append(file_path)
                    fh.write(file_path)
                    fh.write("\n")

    def record_nparray_in_file(self, html_feature_file, no_html_feature_file, np_array):
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
                line_feature = '{} {} # {}\n'.format(label, feature_msg, comments)
                file_open.write(line_feature)
                line_num += 1

    def build_train_test_set(self):
        parent_dir, filename = os.path.split(self.no_html_feature_file)
        filename_wo_ext, ext = os.path.splitext(filename)
        no_html_part_a = os.path.join(parent_dir,
                                      filename_wo_ext + '_random_{}_group_a{}'.format(self.no_html_split_rate, ext))
        no_html_part_b = os.path.join(parent_dir,
                                      filename_wo_ext + '_random_{}_group_b{}'.format(self.no_html_split_rate, ext))

        parent_dir, filename = os.path.split(self.html_feature_file)
        filename_wo_ext, ext = os.path.splitext(filename)
        html_part_a = os.path.join(parent_dir,
                                   filename_wo_ext + '_random_{}_group_a{}'.format(self.html_split_rate, ext))
        html_part_b = os.path.join(parent_dir,
                                   filename_wo_ext + '_random_{}_group_b{}'.format(self.html_split_rate, ext))

        self.train_file = os.path.join(self.train_file_path, 'training_set.txt')
        self.test_file = os.path.join(self.test_file_path, 'test_set.txt')

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

    def train_and_score(self):
        working_dir = r"/sa/githubee/md_auto_tools/src/machine_learning/training_process"
        print '[Change Working Dir] ' + working_dir
        os.chdir(working_dir)

        cmd = 'python train_html.py {}'.format(self.train_file)
        os.system(cmd)
        cmd = 'python predict_html.py {}'.format(self.train_file)
        os.system(cmd)
        cmd = 'python predict_html.py {}'.format(self.test_file)
        os.system(cmd)

    def start_extract(self):
        self.fill_corpus(self.html_sample_path)
        self.html_count = len(self.corpus)
        self.fill_corpus(self.no_html_sample_path)

        hv = HashingVectorizer(input='filename', decode_error='ignore', ngram_range=(1, 3), n_features=100)
        hv_array = hv.transform(self.corpus).toarray()
        self.record_nparray_in_file(self.html_feature_file, self.no_html_feature_file, hv_array)

        self.split_file()

        self.build_train_test_set()

        self.train_and_score()
        self.save_model()

if __name__ == "__main__":
    extract_by_hv = ExtractByHv()
    extract_by_hv.start_extract()