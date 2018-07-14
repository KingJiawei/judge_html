import os,sys

def get_sample(input_file):
    if os.path.isfile(input_file):
        file_open = open(input_file, 'rb')
        for line in file_open:
            if line is not None:
                sample_path_temp = line.split('@')[0].strip()
                sample_path = sample_path_temp.split('#')[-1].strip()
                sample_set.add(sample_path)

def get_feature(feature_file,output_file):
    with open(output_file,'wb') as file_write:
        file_read = open(feature_file, 'rb')
        for line in file_read:
            sample_path_temp = line.split('@')[0].strip()
            sample_path = sample_path_temp.split('#')[-1].strip()
            if sample_path in sample_set:
                file_write.write(line)
                file_write.write('\n')

def print_help():
    print "python check_feature.py log_file feature_file dst_file"

if __name__ == "__main__":
    sample_set = set()
    get_sample(sys.argv[1])
    get_feature(sys.argv[2],sys.argv[3])
