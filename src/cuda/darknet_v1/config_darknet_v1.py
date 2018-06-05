#!/usr/bin/python

import os
import sys
import ConfigParser
import sys
import copy

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

DATASETS = [
    # normal
    # {'txt': 'caltech.pedestrians.1K.txt', 'gold': 'gold.caltech.1K.csv', 'mode': 'full'},
    # {'txt': 'urban.street.1.1K.txt', 'gold': 'gold.urban.street.1.1K.csv', 'mode': 'full'},
    # {'txt': 'voc.2012.1K.txt', 'gold': 'gold.voc.2012.1K.csv', 'mode': 'full'},

    # average
    {'txt': 'caltech.pedestrians.100.txt', 'gold': 'gold.caltech.100.csv', 'mode': 'average'},
    {'txt': 'urban.street.100.txt', 'gold': 'gold.urban.street.100.csv', 'mode': 'average'},
    {'txt': 'voc.2012.100.txt', 'gold': 'gold.voc.2012.100.csv', 'mode': 'average'},

    # very_small for X1 and X2
    # {'txt': 'caltech.pedestrians.10.txt', 'gold': 'gold.caltech.10.csv', 'mode': 'small'},
    # {'txt': 'urban.street.10.txt', 'gold': 'gold.urban.street.10.csv', 'mode': 'small'},
    # {'txt': 'voc.2012.10.txt', 'gold': 'gold.voc.2012.10.csv', 'mode': 'small'},
]

BINARY_NAME = "darknet_v1"
DEBUG_MODE = False
SAVE_LAYER = [0, 1]
USE_TENSOR_CORES = [0, 1]
# 0 - "none",  1 - "gemm", 2 - "smart_pooling", 3 - "l1", 4 - "l2", 5 - "trained_weights"}
ABFT = [0]  # , 2]


def download_weights(src_dir, data_dir):
    os.chdir(data_dir)
    if os.system("./get_darknet_weights.sh") != 0:
        print "ERROR on downloading darknet v1/v2 weights"
        exit(-1)

    os.chdir(src_dir)


def config(board):
    print "Generating darknet for CUDA, board:" + board

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)

        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    data_path = install_dir + "data/darknet"
    bin_path = install_dir + "bin"
    src_darknet = install_dir + "src/cuda/" + BINARY_NAME

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    # executing weights test first
    download_weights(src_dir=src_darknet, data_dir=data_path)

    generate = ["sudo mkdir -p " + bin_path, "mkdir -p /var/radiation-benchmarks/data", "cd " + src_darknet,
                "make clean GPU=1", "make -j 4 GPU=1 SAFE_MALLOC=1",
                "mv ./" + BINARY_NAME + " " + bin_path + "/"]
    execute = []

    for save_layer in SAVE_LAYER:
        for abft in ABFT:
            for i in DATASETS:
                for tc in USE_TENSOR_CORES:
                    if (save_layer == 1 and i['mode'] == 'full') or (save_layer == 0 and i['mode'] == 'small'):
                        continue

                    gold = data_path + '/' + BINARY_NAME + '_' + i['gold']
                    txt_list = install_dir + 'data/networks_img_list/' + i['txt']
                    gen = {
                        'bin': ["sudo " + bin_path, "/" + BINARY_NAME],
                        'e': [' -e ', 'yolo'],  # execution_type =
                        'aa': ['test_radiation', ''],
                        'm': [' -m ', 'valid'],  # execution_model =
                        'c': [' -c ', data_path + '/yolo_v1.cfg'],  # config_file =
                        'w': [' -w ', data_path + '/yolo_v1.weights'],  # weights =
                        'n': [' -n ', '1'],
                        'g': [' -g ', gold],
                        'l': [' -l ', txt_list],
                        'b': [' -b ', src_darknet],
                        'x': [' -x ', 0],
                        's': [' -s ', save_layer],
                        'a': [' -a ', abft],
                        't': [' -t ', tc],
                    }

                    exe = copy.deepcopy(gen)
                    exe['n'][1] = 10000
                    exe['g'][0] = ' -d '

                    exe_save = copy.deepcopy(exe)
                    exe_save['s'][1] = save_layer

                    if abft == 0:
                        generate.append(" ".join([''.join(map(str, value)) for key, value in gen.iteritems()]))

                    execute.append(" ".join([''.join(map(str, value)) for key, value in exe.iteritems()]))

    generate.append("make clean GPU=1 SAFE_MALLOC=1")
    generate.append("make -C ../../include/")
    generate.append("make -j 4 GPU=1 SAFE_MALLOC=1 LOGS=1")
    generate.append("mv ./" + BINARY_NAME + " " + bin_path + "/")

    execute_and_write_json_to_file(execute=execute, generate=generate, install_dir=install_dir,
                                   benchmark_bin=BINARY_NAME, debug=DEBUG_MODE)

if __name__ == "__main__":
    global DEBUG_MODE

    parameter = str(sys.argv[1:]).upper() 
    if parameter == 'DEBUG':
        DEBUG_MODE = True

    board, _ = discover_board()
    config(board=board)
