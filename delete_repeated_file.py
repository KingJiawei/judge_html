import sys
import os

if __name__ == "__main__":
    src_path = sys.argv[1]
    target_path = sys.argv[2]
    file_set = set()
    delete_num = 0
    if os.path.isdir(src_path):
        for root, dirs, files in os.walk(src_path):
            for name in files:
                file_set.add(name)
                #file_path = os.path.join(root, name)
    print "{} have exist!".format(len(file_set))
    if os.path.isdir(target_path):
        for root, dirs, files in os.walk(target_path):
            for name in files:
                if name in file_set:
                    delete_num += 1
                    file_path = os.path.join(root, name)
                    print "delete {}".format(file_path)
                    os.remove(file_path)
        print "delete {} files!".format(delete_num)