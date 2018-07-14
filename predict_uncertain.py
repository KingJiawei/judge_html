import os
import sys
import time
import json
from sklearn.datasets import load_svmlight_file
from classifier_svm import SVMClassifier
from classifier_xgboost import TMXGBClassifier
if os.name != 'nt':
    from classifier_keras import TMSAKerasClassifier
    from classifier_pylibsvm import LIBSVMClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import multiprocessing


class TMXGBClassifier:
    def __init__(self, config):
        self.config_ = config
        self.model_path_ = ''
        self.model_ = None

    def load_model(self, model_path = None):
        if model_path and os.path.exists(model_path):
            print('Load model from parameter: {}'.format(model_path))
            self.model_path_ = model_path
        else:
            self.model_path_ = os.path.join(self.config_['filesystem']['mid_folder'], \
                                            self.config_['model']['xgboost']['model_name'])
            print('Load model from config: {}'.format(self.model_path_))
        #
        thread_num = self.config_['model']['xgboost']['nthread']
        if self.config_['model']['xgboost']['use_system_cpu_num']:
            thread_num = multiprocessing.cpu_count()
        self.model_ = xgb.Booster({'nthread': thread_num})
        self.model_.load_model(self.model_path_)

    def score(self, X, y):
        try:
            testing_set = xgb.DMatrix(X)
            y_pred = self.model_.predict(testing_set)
            return y_pred
        except Exception, e:
            print('[XGBClassifier.score] Exception: {}'.format(str(e)))


class ClassifierFactory:
    """"""
    def __init__(self, config):
        self.config_ = config

    def produce(self, model_name):
        if 'svm' == model_name:
            classifier = SVMClassifier(self.config_)
            classifier.load_model()
            return classifier
        elif 'xgboost' == model_name:
            classifier = TMXGBClassifier(self.config_)
            classifier.load_model()
            return classifier
        elif 'libsvm' == model_name:
            if os.name != 'nt':
                classifier = LIBSVMClassifier(self.config_)
                classifier.load_model()
                return classifier
            else:
                return None
        elif 'keras' == model_name:
            if os.name != 'nt':
                classifier = TMSAKerasClassifier(self.config_)
                classifier.load_model()
                return classifier
            else:
                return None
        else:
            print '[ERROR] Unsupported Model Name'

class ClassifierHelper:
    def __init__(self, config,path):
        self.config_ = config
        self.classifier_factory_ = ClassifierFactory(config)
        #self.predict_root_path = "/home/jiawei/judge_html/test_doc2vec/"
        self.predict_root_path = path
        self.predict_0_1 = self.predict_root_path + "0_1"
        self.predict_1_2 = self.predict_root_path + "1_2"
        self.predict_2_3 = self.predict_root_path + "2_3"
        self.predict_3_4 = self.predict_root_path + "3_4"
        self.predict_4_5 = self.predict_root_path + "4_5"
        self.predict_5_6 = self.predict_root_path + "5_6"
        self.predict_6_7 = self.predict_root_path + "6_7"
        self.predict_7_8 = self.predict_root_path + "7_8"
        self.predict_8_9 = self.predict_root_path + "8_9"
        self.predict_9_10 = self.predict_root_path + "9_10"
        self.predict_dir_set = [self.predict_0_1,self.predict_1_2,self.predict_2_3,\
                            self.predict_3_4,self.predict_4_5,self.predict_5_6, \
                            self.predict_6_7,self.predict_7_8,self.predict_8_9, \
                            self.predict_9_10]
        for dir in self.predict_dir_set:
            if not os.path.exists(dir):
                os.mkdir(dir)
                os.chmod(dir, 777)

        self.y_pred = []


    def score(self, testing_file):
        try:
            model_name_list = self.config_['train']['model'].split('|')
            print '[Prediction Process] model list: {}'.format(model_name_list)
            print '> load testing file: {}'.format(testing_file)
            X, y = load_svmlight_file(testing_file)
            print '- load complete'
            for model_name in model_name_list:
                print '> calculate scores using [{}] model'.format(model_name)
                classifier = self.classifier_factory_.produce(model_name)
                start = time.time()
                self.y_pred = classifier.score(X,y)
                msg = "Scoring Time Delta: {}".format(time.time() - start)
                print(msg)

                if model_name == 'svm':
                    model_name += '_{}'.format(self.config_['model']['svm']['kernel'])
        except Exception, e:
            print '[ClassifierHelper.score] Exception: {}'.format(str(e))
        print 'Done!'

    def predict(self,input_file):
        if os.path.isfile(input_file):
            file_open = open(input_file, 'rb')
            line_count = 0
            for line in file_open:
                if line is not None:
                    sample_path_temp = line.split('@')[0].strip()
                    sample_path = sample_path_temp.split('#')[-1].strip()
                    if self.y_pred[line_count] >= 0.9:
                        cmd = "cp {} {}".format(sample_path, self.predict_9_10)
                    elif self.y_pred[line_count] >= 0.8:
                        cmd = "cp {} {}".format(sample_path, self.predict_8_9)
                    elif self.y_pred[line_count] >= 0.7:
                        cmd = "cp {} {}".format(sample_path, self.predict_7_8)
                    elif self.y_pred[line_count] >= 0.6:
                        cmd = "cp {} {}".format(sample_path, self.predict_6_7)
                    elif self.y_pred[line_count] >= 0.5:
                        cmd = "cp {} {}".format(sample_path, self.predict_5_6)
                    elif self.y_pred[line_count] >= 0.4:
                        cmd = "cp {} {}".format(sample_path, self.predict_4_5)
                    elif self.y_pred[line_count] >= 0.3:
                        cmd = "cp {} {}".format(sample_path, self.predict_3_4)
                    elif self.y_pred[line_count] >= 0.2:
                        cmd = "cp {} {}".format(sample_path, self.predict_2_3)
                    elif self.y_pred[line_count] >= 0.1:
                        cmd = "cp {} {}".format(sample_path, self.predict_1_2)
                    else:
                        cmd = "cp {} {}".format(sample_path, self.predict_0_1)
                    print cmd
                    os.system(cmd)
                    line_count += 1


help_msg = """
Usage:
    > python predict_uncertain.py testing-file dst_dir
"""

if __name__ == '__main__':
    with open('config_html.json', 'rb') as fh:
        config = json.load(fh)
    helper = ClassifierHelper(config,sys.argv[2])
    helper.score(sys.argv[1])
    helper.predict(sys.argv[1])

