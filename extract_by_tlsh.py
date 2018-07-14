import os,sys
import tlsh
import collections
import shutil
import time

class UseTlsh:
    def __init__(self):
        self.no_html_feature_file = "/home/jiawei/judge_html/test_tlsh/no_html_feature_file.txt"
        self.html_feature_file = "/home/jiawei/judge_html/test_tlsh/html_feature_file.txt"
        self.no_html_split_rate = 0.7
        self.html_split_rate = 0.7
        self.html_sample_path = "/home/jiawei/judge_html/old_html_file"
        self.no_html_sample_path = "/home/jiawei/judge_html/old_no_html_file"
        self.train_file_path = "/home/jiawei/judge_html/test_tlsh/"
        self.test_file_path = "/home/jiawei/judge_html/test_tlsh/"
        self.train_file = None
        self.test_file = None
        self.model_path = "/sa/middle_dir/judge_html"


    def compute_1(self,path):
        with open(path, 'rb') as f:
            data = f.read()
            hs = tlsh.forcehash(data)
        return hs

    def compute_2(self,path):
        h = tlsh.Tlsh()
        with open(path, 'rb') as f:
            for buf in iter(lambda: f.read(512), b''):
                h.update(buf)
        h.final()
        return h.hexdigest()

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

    def extract_feature(self,file_path):
        file_tlsh = self.compute_1(file_path)
        file_tlsh_list = list(file_tlsh.strip())
        file_tlsh_dict = {}
        for i in range(0,len(file_tlsh_list)):
            file_tlsh_dict[i] = int(file_tlsh_list[i],16)
        return file_tlsh_dict

    def extract_label(self,file_path):
        normal_num = file_path.count('normal')
        malicious_num = file_path.count('malicious')
        if normal_num < malicious_num:
            label = 1
        else:
            label = 0
        return label

    def dump_feature(self,label,src_path,dst_path):
        with open(dst_path, 'w') as output:
            if os.path.isdir(src_path):
                for root, dirs, files in os.walk(src_path):
                    for name in files:
                        file_path = os.path.join(root, name)
                        try:
                            features = self.extract_feature(file_path)
                            feature_line = self.convert_to_libsvm_format(label, features, file_path)
                            output.write(feature_line)
                        except Exception,e:
                            print '[ERROR] cannot extract feature on {}, exception is {}'.format(file_path, str(e))

            elif os.path.isfile(src_path):
                feature_msg = self.extract_feature(src_path)
                output.write(self.convert_to_libsvm_format(label, feature_msg,src_path))
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
        print "dump {}".format(self.html_sample_path)
        self.dump_feature(1,self.html_sample_path,self.html_feature_file)
        print "dump {}".format(self.no_html_sample_path)
        self.dump_feature(0,self.no_html_sample_path,self.no_html_feature_file)

        self.split_file()

        self.build_train_test_set()

        self.train_and_score()
        self.save_model()




def print_help():
    print """
python computer_tlsh.py 
    """

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print_help()
    else:
        use_tlsh = UseTlsh()
        use_tlsh.start()
