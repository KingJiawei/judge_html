import os
import sys

def start_extract(path,file):
    with open(file, 'wb') as file_open:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for name in files:
                    file_open.write(name)
                    file_open.write('\n')

        else:
            print "please input a path!"


if __name__ == "__main__":
    start_extract(sys.argv[1],sys.argv[2])