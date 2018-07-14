import re
import sys
import os
import shutil
import csv

class JudgeByVt:
    def __init__(self):
        self.html_path = './html_file'
        self.html_script_path = './html_file/script/'
        self.html_html_script_path = './html_file/html_script/'
        self.html_html_head_path = './html_file/html_head/'
        self.html_html_body_path = './html_file/html_body/'
        self.html_doctype_html_path = './html_file/doctype_html/'
        self.html_doctype_path = './html_file/doctype/'
        self.html_judge_by_vt_path = './html_file/judge_by_vt'

        self.no_html_path = './no_html_file'
        self.no_html_judge_by_vt_path = './no_html_file/judge_by_vt'
        self.no_html_no_match_by_re_path = './no_html_file/no_match_by_re'

        self.uncertain_path = './uncertain_file'
        self.uncertain_no_type_path = './uncertain_file/no_type'
        self.uncertain_no_file_path = './uncertain_file/no_file'

        self.dir_list = [self.html_path,self.html_script_path,self.html_judge_by_vt_path,\
                         self.html_html_script_path, self.html_doctype_html_path,self.html_doctype_path, \
                         self.html_html_head_path,self.html_html_body_path,\
                         self.no_html_path, self.no_html_judge_by_vt_path,self.no_html_no_match_by_re_path, \
                         self.uncertain_path,self.uncertain_no_type_path,self.uncertain_no_file_path]

        self.sha1_name = None
        self.script_count = 0

    # if file don't have html or script string,I say it isn't a html file
    def judge_html_by_re(self,content):
#        p1 = re.compile(r'[\s\S]*?<!DOCTYPE html', re.I)
        p1 = re.compile(r'(?<!\"|\'|/)<!DOCTYPE html', re.I | re.U)
        s1 = p1.search(content)
        if s1:
            print s1.group()
            p12 = re.compile(r'(?<!\"|\'|/)<html.*?>', re.I | re.U)
            s12 = p12.search(content)
            if s12:
                print s12.group()
                print 12
                return 12
            else:
                print 1
                return 1
        else:
#            p2 = re.compile(r'[\s\S]*?<html[\s]*>', re.I)
            p2 = re.compile(r'(?<!\"|\'|/)<html.*?>', re.I | re.U)
            s2 = p2.search(content)
            if s2:
                print s2.group()
                p22 = re.compile(r'<script.*>.*</script>', re.I | re.U|re.DOTALL)
                s22 = p22.search(content)
                if s22:
                    #print s22.group()
                    print 22
                    return 22

                p23 = re.compile(r'<head.*>.*</head>', re.I | re.U|re.DOTALL)
                s23 = p23.search(content)
                if s23:
                    #print s23.group()
                    print 23
                    return 23

                p24 = re.compile(r'<body.*>.*</body>', re.I | re.U|re.DOTALL)
                s24 = p24.search(content)
                if s24:
                    #print s24.group()
                    print 24
                    return 24
                print 2
                return 2

            else:
                p3 = re.compile(r'\s*<script.*?>.+</script>\s*', re.I | re.U|re.DOTALL)
                s3 = p3.match(content)
                if s3:
                    print s3.group()
                    print 3
                    return 3
                else:
                    print 0
                    return 0

    #after judge by re,check it in Vt ,if I can't find its SHA1 or type,
    #I say it's a html,otherwise I dont't think so
    def judge_by_vt(self,file_sha1,file,reader):
        self.file_type = None
        file_to_read = open(file, 'rb')
        content = file_to_read.read()
        result_by_re = self.judge_html_by_re(content)
        if result_by_re:
            if result_by_re == 3:
                self.script_count += 1
                shutil.copy2(file, self.html_script_path)
            elif result_by_re == 22:
                shutil.copy2(file, self.html_html_script_path)
            elif result_by_re == 23:
                shutil.copy2(file, self.html_html_head_path)
            elif result_by_re == 24:
                shutil.copy2(file, self.html_html_body_path)
            elif result_by_re == 12:
                shutil.copy2(file, self.html_doctype_html_path)
            elif result_by_re == 1:
                shutil.copy2(file, self.html_doctype_path)
            else:
                print "check VtResult!"
                sha1_in_csv_flag = 0
                for row in reader:
                    if row['SHA1'] == file_sha1:
                        sha1_in_csv_flag = 1
                        print row['File Type']
                        if len(row['File Type']) == 0:
                            print "can't find file type in vt!"
                            shutil.copy2(file, self.uncertain_no_type_path)
                        elif row['File Type'] != 'HTML':
                            print "vt said file type isn't html!"
                            shutil.copy2(file, self.no_html_judge_by_vt_path)
                        else:
                            print "vt said file type is html"
                            shutil.copy2(file, self.html_judge_by_vt_path)
                if sha1_in_csv_flag == 0:
                    print "can't find this file in vt!"
                    shutil.copy2(file, self.uncertain_no_file_path)

        else:
            print "no match by re!"
            shutil.copy2(file, self.no_html_no_match_by_re_path)

    def judge_by_vt_single(self,file_sha1,file,reader):
        self.file_type = None
        file_to_read = open(file, 'rb')
        content = file_to_read.read()
        result_by_re = self.judge_html_by_re(content)
        if result_by_re:
            if result_by_re == 3:
                pass
            elif result_by_re == 22:
                pass
            elif result_by_re == 23:
                pass
            elif result_by_re == 24:
                pass
            elif result_by_re == 12:
                pass
            elif result_by_re == 1:
                pass
            else:
                print "check VtResult!"
                sha1_in_csv_flag = 0
                for row in reader:
                    if row['SHA1'] == file_sha1:
                        sha1_in_csv_flag = 1
                        print row['File Type']
                        if len(row['File Type']) == 0:
                            print "can't find file type in vt!"
                        elif row['File Type'] != 'HTML':
                            print "vt said file type isn't html!"
                            return
                        else:
                            print "vt said file type is html"
                if sha1_in_csv_flag == 0:
                    print "can't find this file in vt!"

        else:
            print "no match by re!"

    def make_all_dir(self,dir_list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.mkdir(dir)
                os.chmod(dir, 777)

    def start_judge(self,src_path):
        self.make_all_dir(self.dir_list)
        sample_number = 0
        with open('VtResult.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            if os.path.isdir(src_path):
                for root, dirs, files in os.walk(src_path):
                    for name in files:
                        file_path = os.path.join(root, name)
                        sample_number += 1
                        print "\n"
                        print "###############  {}  ###############".format(sample_number)
                        self.judge_by_vt(name, file_path,reader)
            elif os.path.isfile(src_path):
                name = src_path.split('\ | /')[-1]
                file_path = src_path
                self.judge_by_vt(name, file_path,reader)
            else:
                pass

if __name__ == "__main__":
    judge_by_vt = JudgeByVt()
    judge_by_vt.start_judge(sys.argv[1])
