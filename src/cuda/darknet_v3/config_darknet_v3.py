#!/usr/bin/python
import os
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
    # {'txt': 'urban.street.100.txt', 'gold': 'gold.urban.street.100.csv', 'mode': 'average'},
    # {'txt': 'voc.2012.100.txt', 'gold': 'gold.voc.2012.100.csv', 'mode': 'average'},

    # very_small for X1 and X2
    # {'txt': 'caltech.pedestrians.10.txt', 'gold': 'gold.caltech.10.csv', 'mode': 'small'},
    # {'txt': 'urban.street.10.txt', 'gold': 'gold.urban.street.10.csv', 'mode': 'small'},
    # {'txt': 'voc.2012.10.txt', 'gold': 'gold.voc.2012.10.csv', 'mode': 'small'},
]

BINARY_NAME = "darknet_v3"
# SAVE_LAYER = [0, ]
USE_TENSOR_CORES = [0]#, 1]
# 0 - "none",  1 - "gemm", 2 - "smart_pooling", 3 - "l1", 4 - "l2", 5 - "trained_weights"}
ABFT = [0]  # , 2]
REAL_TYPES = ["single", "half"]
WEIGHTS = "yolov3.weights"
CFG = "yolov3.cfg"


def config(board, debug):
    print "Generating darknet v3 for CUDA, board:" + board

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    benchmark_bin = BINARY_NAME
    data_path = install_dir + "data/darknet"
    bin_path = install_dir + "bin"
    src_darknet = install_dir + "src/cuda/" + benchmark_bin

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    # change it for darknetv3
    generate = ["mkdir -p " + bin_path, "mkdir -p /var/radiation-benchmarks/data", "cd " + src_darknet,
                "make -C ../../include/"]
    execute = []

    execute_and_write_json_to_file(execute=execute, generate=generate, install_dir=install_dir,
                                   benchmark_bin=benchmark_bin, debug=debug)

    # 0 - "none",  1 - "gemm", 2 - "smart_pooling", 3 - "l1", 4 - "l2", 5 - "trained_weights"}
    for fp_precision in REAL_TYPES:
        for i in DATASETS:
            for tc in USE_TENSOR_CORES:
                generate = ["make clean GPU=1 LOGS=1"]
                execute = []
                bin_final_name = benchmark_bin + "_" + fp_precision
                generate.append("make -j4 LOGS=1 GPU=1 REAL_TYPE=" + fp_precision)
                generate.append("mv ./" + bin_final_name + "  " + bin_path + "/")

                gold = data_path + '/' + BINARY_NAME + '_tensor_cores_mode_' + str(tc) + '_fp_precision_' + str(
                    fp_precision) + '_' + i['gold']
                txt_list = install_dir + 'data/networks_img_list/' + i['txt']
                exec_bin_path = "sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
                exec_bin_path += " " + bin_path + "/" + bin_final_name

                gen = [None] * 10
                gen[0] = [exec_bin_path]
                gen[1] = [" detector test_radiation "]
                gen[2] = [src_darknet + "/cfg/coco.data"]
                gen[3] = [data_path + '/' + CFG]
                gen[4] = [data_path + '/' + WEIGHTS]
                gen[5] = [txt_list]
                gen[6] = [' -generate ', '1']
                gen[7] = [' -iterations ', '1']
                gen[8] = [' -tensor_cores ', str(tc)]
                gen[9] = [' -gold ', gold]

                generate.append("make -j 4 GPU=1 LOGS=1 REAL_TYPE=" + fp_precision)
                generate.append("mv ./" + bin_final_name + "  " + bin_path + "/")

                exe = copy.deepcopy(gen)
                exe[7][1] = '1000000'
                exe[6][1] = '0'

                generate.append(" ".join([''.join(value) for value in gen]))

                execute.append(" ".join([''.join(value) for value in exe]))

                execute_and_write_json_to_file(execute=execute, generate=generate, install_dir=install_dir,
                                               benchmark_bin=bin_final_name, debug=debug)


if __name__ == "__main__":
    debug_mode = False
    download_data = False
    try:
        parameter = str(sys.argv[1:][0]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except Exception as err:
        debug_mode = False
        download_data = False

    board, _ = discover_board()
    config(board=board, debug=debug_mode)
