import os,sys
import re
import collections
import time
import shutil
import yara

class ExtractByKeywordYara:
    def __init__(self):
        self.keyword_file = "keyword.yar"
        self.feature_list = []
        self.feature_dict = {}
        self.html_pattern_list = []

        self.no_html_feature_file = "/home/jiawei/judge_html/test_keyword_by_yara/no_html_feature_file.txt"
        self.html_feature_file = "/home/jiawei/judge_html/test_keyword_by_yara/html_feature_file.txt"
        self.no_html_split_rate = 0.7
        self.html_split_rate = 0.7
        self.html_sample_path = "/home/jiawei/judge_html/html_file"
        self.no_html_sample_path = "/home/jiawei/judge_html/no_html_file"
        self.uncertain_path = "/home/jiawei/judge_html/uncertain_file"
        self.uncertain_feature_file = "/home/jiawei/judge_html/test_keyword_by_yara/uncertain_feature_file.txt"

        self.train_file_path = "/home/jiawei/judge_html/test_keyword_yara/"
        self.test_file_path = "/home/jiawei/judge_html/test_keyword_by_yara/"
        self.train_file = None
        self.test_file = None
        self.model_path = "/sa/middle_dir/judge_html"

        self.yara_rules_ = yara.compile(self.keyword_file)
        self.matched_feature_index_map_ = {}
        self.total_num = 0

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

    def dump_feature(self, label, src_path, dst_path):
        with open(dst_path, 'wb') as output:
            if os.path.isdir(src_path):
                for root, dirs, files in os.walk(src_path):
                    for name in files:
                        file_path = os.path.join(root, name)
                        try:
                            self.total_num += 1
                            file_to_read = open(file_path, 'rb')
                            content = file_to_read.read()
                            features = self.analyze_content(content)
                            feature_line = self.convert_to_libsvm_format(label, features, file_path)
                            output.write(feature_line)
                            print "############ {} {}  ################".format(self.total_num,file_path)
                        except Exception, e:
                            print '[ERROR] cannot extract feature on {}, exception is {}'.format(file_path, str(e))
            elif os.path.isfile(src_path):
                name = src_path.split('\ | /')[-1]
                file_path = src_path
                try:
                    file_to_read = open(file_path, 'rb')
                    content = file_to_read.read()
                    features = self.analyze_content(content)
                    feature_line = self.convert_to_libsvm_format(label, features, file_path)
                    output.write(feature_line)
                    # print "extract feature on {}".format(file_path)
                except Exception, e:
                    print '[ERROR] cannot extract feature on {}, exception is {}'.format(file_path, str(e))
            else:
                pass

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

    def start(self):
        print "dump {}".format(self.uncertain_path)
        self.dump_feature(0, self.uncertain_path, self.uncertain_feature_file)


if __name__ == "__main__":
    extract_by_keyword_yar = ExtractByKeywordYara()
    extract_by_keyword_yar.start()