#!/usr/bin/python

import os
import sys
import ConfigParser
import copy

DATASETS = [
    # normal
    {'txt': 'caltech.pedestrians.1K.txt', 'gold': 'gold.caltech.1K.test', 'mode': 'full'},
    {'txt': 'urban.street.1.1K.txt', 'gold': 'gold.urban.street.1.1K.test', 'mode': 'full'},
    {'txt': 'voc.2012.1K.txt', 'gold': 'gold.voc.2012.1K.test', 'mode': 'full'},

    # very_small for X1 and X2
    {'txt': 'caltech.pedestrians.10.txt', 'gold': 'gold.caltech.10.test', 'mode': 'small'},
    {'txt': 'urban.street.10.txt', 'gold': 'gold.urban.street.10.test', 'mode': 'small'},
    {'txt': 'voc.2012.10.txt', 'gold': 'gold.voc.2012.10.test', 'mode': 'small'},
]

#full
#WEIGHTS="vgg16"
WEIGHTS="zf"

def main(board):
    print "Generating py-faster for CUDA on " + str(board)

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)

        installDir = config.get('DEFAULT', 'installdir') + "/"
        # varDir = config.get('DEFAULT', 'vardir') + "/"
        # logDir = config.get('DEFAULT', 'logdir') + "/"
        # tmpDir = config.get('DEFAULT', 'tmpdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    benchmark_bin = "py_faster_rcnn.py"
    data_path = installDir + "data/py_faster_rcnn"
    # bin_path = installDir + "bin"
    src_py_faster = installDir + "src/cuda/py-faster-rcnn"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    generate = [str("cd " + src_py_faster), "make -C " + installDir + "/src/include/log_helper_swig_wraper/ log_helper_python"]
    execute = []

    for i in DATASETS:
        if board in ['K1', 'X1', 'X2'] and i['mode'] == 'full':
            continue

        gold = data_path + '/py_faster_' + i['gold']
        txt_list = installDir + 'data/networks_img_list/' + i['txt']
        gen = {
            'gold': [' --gen ', gold],
            'iml': [' --iml ', txt_list],
            'ite': [' --ite ', '1000'],
            'zexe': ['sudo python ', src_py_faster + "/tools/" + benchmark_bin + " "],
            'net': ['--net ', WEIGHTS]
        }

        exe = copy.deepcopy(gen)
        exe['gold'][0] = ' --gld '
        exe['log'] = [' --log ', ' daniel_logs ']

        generate.append(" ".join([''.join(map(str, value)) for key, value in gen.iteritems()]))
        execute.append(" ".join([''.join(map(str, value)) for key, value in exe.iteritems()]))

        execute_and_write_how_to_file(execute, generate, installDir, benchmark_bin)

    # os.system("cd " + src_py_faster)
    # for i in generate:
    #     if os.system(str(i)) != 0:
    #         print "Something went wrong with generate of ", str(i)
    #         exit(1)
    # #print i
    #
    # fp = open(installDir + "scripts/how_to_run_py_faster_rcnn_cuda_" + board, 'w')
    #
    # for i in execute:
    #     print >> fp, "[\"" + str(i) + "\" , 0.016, \"py_faster_rcnn.py\"],"
    #     print "[\"" + str(i) + "\" , 0.016, \"py_faster_rcnn.py\"],"
    #
    # print "\nConfiguring done, to run check file: " + installDir + "scripts/how_to_run_py_faster_rcnn_cuda_" + str(
    #     board) + "\n"
    #
    # sys.exit(0)


def execute_and_write_how_to_file(execute, generate, installDir, benchmark_bin):
    for i in generate:
        if os.system(str(i)) != 0:
            print "Something went wrong with generate of ", str(i)
            exit(1)
        print i
    fp = open(installDir + "scripts/json_files/" + benchmark_bin + ".json", 'w')

    list_to_print = ["["]
    for ii, i in enumerate(execute):
        command = "{\"killcmd\": \"pkill -9 " + benchmark_bin + "\", \"exec\": \"" + str(i) + "\"}"
        if ii != len(execute) - 1:
            command += ', '
        list_to_print.append(command)
    list_to_print.append("]")

    for i in list_to_print:
        print >> fp, i
        print i
    fp.close()
    print "\nConfiguring done, to run check file: " + installDir + "scripts/json_files/" + benchmark_bin + ".json"


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/k40>"
    else:
        main(str(parameter[0]).upper())
