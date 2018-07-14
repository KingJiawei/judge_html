import os
import sys
import time
import json
from file_helper import *
from sklearn.datasets import load_svmlight_file
from classifier_svm import SVMClassifier
from classifier_xgboost import TMXGBClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
if os.name != 'nt':
    from classifier_keras import TMSAKerasClassifier
    from classifier_pylibsvm import LIBSVMClassifier
import multiprocessing

class TMXGBClassifierHtml:
    def __init__(self, config):
        self.config_ = config
        self.model_path_ = ''
        self.model_ = None
        self.tpl_ = []
        self.tnl_ = []
        self.fpl_ = []
        self.fnl_ = []
        self.tp_ = 0
        self.tn_ = 0
        self.fp_ = 0
        self.fn_ = 0

    def clear_score(self):
        self.tpl_ = []
        self.tnl_ = []
        self.fpl_ = []
        self.fnl_ = []
        self.tp_ = 0
        self.tn_ = 0
        self.fp_ = 0
        self.fn_ = 0

    def load_model(self, model_path=None):
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

    def score(self, X, y,threshold=0.5):
        try:
            self.clear_score()
            testing_set = xgb.DMatrix(X)
            y_pred = self.model_.predict(testing_set)
            for i in range(0,y_pred.size):
                if y_pred[i] >= threshold and y[i] == 1:
                    self.tp_ += 1
                    self.tpl_.append(i)
                elif y_pred[i] < threshold and y[i] == 0:
                    self.tn_ += 1
                    self.tnl_.append(i)
                elif y_pred[i] >= threshold and y[i] == 0:
                    self.fp_ += 1
                    self.fpl_.append(i)
                else:
                    self.fn_ += 1
                    self.fnl_.append(i)
            return self.tpl_, self.tnl_, self.fpl_, self.fnl_, y_pred
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
            classifier = TMXGBClassifierHtml(self.config_)
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
    def __init__(self, config):
        self.config_ = config
        self.classifier_factory_ = ClassifierFactory(config)

    def score(self, testing_file,threshold):
        try:
            file_helper = FileHelper()
            model_name_list = self.config_['train']['model'].split('|')
            print '[Prediction Process] model list: {}'.format(model_name_list)
            print '> load testing file: {}'.format(testing_file)
            X, y = load_svmlight_file(testing_file)
            print '- load complete'
            for model_name in model_name_list:
                test_filename = os.path.splitext(os.path.split(testing_file)[1])[0]
                print '> calculate scores using [{}] model'.format(model_name)
                tpl = tnl = fpl = fnl = []
                classifier = self.classifier_factory_.produce(model_name)
                start = time.time()
                (tpl, tnl, fpl, fnl, y_pred) = classifier.score(X,y,threshold)
                msg = "Scoring Time Delta: {}".format(time.time() - start)
                print(msg)
                #
                tp = len(tpl)
                tn = len(tnl)
                fp = len(fpl)
                fn = len(fnl)
                msg = "TP:{}, TN:{}, FP:{}, FN:{}".format(tp, tn, fp, fn)
                print(msg)
                #
                accuracy = float(tp+tn)/(tp+tn+fp+fn)
                if 0 == tp+fp:
                    precision = 0
                else:
                    precision = float(tp)/(tp+fp)
                if 0 == tp+fn:
                    recall = 0
                else:
                    recall = float(tp)/(tp+fn)
                if 0 == fp+tn:
                    fpr = 0
                else:
                    fpr = float(fp)/(fp+tn)
                fdr = 1-precision
                f1 = float(2*tp)/(2*tp+fp+fn)
                f2 = 5*precision*recall/(4*precision+recall)
                auc = roc_auc_score(y, y_pred)
                msg = "Accuracy:{}, Precision(PPV):{}, Recall:{}, FPR(FP1):{}, FDR(FP2|1-PPV):{}, F1-Measure:{}, F2-Measure:{}, AUC:{}".format(\
                    accuracy, precision, recall, fpr, fdr, f1, f2, auc)
                print(msg)
                print '> save false prediction result'
                path_wo_ext, ext = os.path.splitext(testing_file)
                file_dir, file_name = os.path.split(path_wo_ext)
                if model_name == 'svm':
                    model_name += '_{}'.format(self.config_['model']['svm']['kernel'])
                fpl_file = os.path.join(self.config_['filesystem']['mid_folder'], '{}_{}_fpl{}'.format(file_name, model_name, ext))
                file_helper.backup_lines_by_index(testing_file, fpl, fpl_file)
                print(fpl_file)
                fnl_file = os.path.join(self.config_['filesystem']['mid_folder'], '{}_{}_fnl{}'.format(file_name, model_name, ext))
                file_helper.backup_lines_by_index(testing_file, fnl, fnl_file)
                print(fnl_file)
        except Exception, e:
            print '[ClassifierHelper.score] Exception: {}'.format(str(e))
        print 'Done!'


help_msg = """
Usage:
    > python predict_html.py testing-file threshold
"""

if __name__ == '__main__':
    with open('config_html.json', 'rb') as fh:
        config = json.load(fh)
    helper = ClassifierHelper(config)
    if len(sys.argv)==2:
        helper.score(sys.argv[1],0.5)
    else:
        helper.score(sys.argv[1],float(sys.argv[2]))
