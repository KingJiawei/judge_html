import gensim
import os,sys
import time
import collections
import shutil
import multiprocessing
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
LabeledSentence = gensim.models.doc2vec.LabeledSentence

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for i, doc in enumerate(self.doc_list):
            #yield LabeledSentence(words=doc.split(),tags=[self.labels_list[i]])
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), tags=[i])

class UseDoc2vec:
    def __init__(self):
        self.data_label = []
        self.data = []
        self.train_data = None
        self.model = None
        self.model_name = None


        self.all_sample_path = "/home/jiawei/judge_html/all_file"
        self.html_sample_path = "/home/jiawei/judge_html/html_file"
        self.no_html_sample_path = "/home/jiawei/judge_html/no_html_file"
        self.no_html_feature_file_path = "/home/jiawei/judge_html/test_doc2vec/"
        self.html_feature_file_path = "/home/jiawei/judge_html/test_doc2vec/"
        self.no_html_feature_file = "/home/jiawei/judge_html/test_doc2vec/no_html_feature_file.txt"
        self.html_feature_file = "/home/jiawei/judge_html/test_doc2vec/html_feature_file.txt"
        self.no_html_split_rate = 0.7
        self.html_split_rate = 0.7
        self.train_file_path = "/home/jiawei/judge_html/test_doc2vec/"
        self.test_file_path = "/home/jiawei/judge_html/test_doc2vec/"
        self.model_path = "/sa/middle_dir/judge_html"

    def process_train_data(self,src_train_path):
        if os.path.isdir(src_train_path):
            for root, dirs, files in os.walk(src_train_path):
                print "preprocess {}".format(root)
                for name in files:
                    file_path = os.path.join(root, name)
                    self.data_label.append(name)
                    self.data.append(open(file_path,'r').read())
        elif os.path.isfile(src_train_path):
            self.data_label.append(src_train_path)
            self.data.append(open(src_train_path, 'r'))
        else:
            pass
        #print self.data
        #print self.data_label
        self.train_data = LabeledLineSentence(self.data,self.data_label)

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


    def extract_feature(self,src_test_path):
        feature = []
        if os.path.isdir(src_test_path):
            print "should be a file!"
            pass
        elif os.path.isfile(src_test_path):
            feature = self.model.infer_vector(gensim.utils.simple_preprocess(open(src_test_path, 'r').read()))
        else:
            pass
        feature_dict = {}
        for i in range(0,len(feature)):
            feature_dict[i] = feature[i]
        return feature_dict

    def dump_feature(self,label,src_path,dst_path):
        with open(dst_path, 'w') as output:
            if os.path.isdir(src_path):
                for root, dirs, files in os.walk(src_path):
                    print "extract feature on {}".format(root)
                    for name in files:
                        file_path = os.path.join(root, name)
                        try:
                            features = self.extract_feature(file_path)
                            feature_line = self.convert_to_libsvm_format(label, features, file_path)
                            output.write(feature_line)
                            #print "extract feature on {}".format(file_path)
                        except Exception,e:
                            print '[ERROR] cannot extract feature on {}, exception is {}'.format(file_path, str(e))

            elif os.path.isfile(src_path):
                feature_msg = self.extract_feature(src_path)
                output.write(self.convert_to_libsvm_format(label, feature_msg,src_path))
            else:
                pass


    def train_doc2vec_model(self):
        self.model = gensim.models.Doc2Vec(size=300, min_count=3, iter=55,workers=multiprocessing.cpu_count())
        self.model.build_vocab(self.train_data)
        self.model.train(self.train_data,total_examples=self.model.corpus_count, epochs=self.model.iter)
        self.model.save("doc2vec_{}.model".format( time.strftime('%Y%m%d%H%M%S', time.localtime())))

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


    def test(self):
        return self.model.infer_vector(['html', 'br', 'body', 'head', 'a', 'div'])


    def start(self):
        if self.model_name is None:
            print "preprocess Data!"
            self.process_train_data(self.all_sample_path)
            print "start train doc2vec model!"
            self.train_doc2vec_model()
        #print self.test()
        #self.extract_feature(self.test_file)
        else:
            self.model = Doc2Vec.load(self.model_name)
        print "dump {}".format(self.html_sample_path)
        self.dump_feature(1, self.html_sample_path, self.html_feature_file)
        print "dump {}".format(self.no_html_sample_path)
        self.dump_feature(0, self.no_html_sample_path, self.no_html_feature_file)


        self.split_file()

        self.build_train_test_set()

        self.train_and_score()
        self.save_model()




def print_help():
    print """
python use_doc2vec.py
    """

if __name__ == "__main__":
    use_doc2vec = UseDoc2vec()
    if len(sys.argv) != 1:
        use_doc2vec.model_name = sys.argv[1]
    else:
        pass
    use_doc2vec.start()

