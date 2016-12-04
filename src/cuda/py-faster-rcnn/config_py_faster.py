#!/usr/bin/python

import os
import sys
import ConfigParser
import copy

DATASETS = [
            {'txt':'caltech.pedestrians.critical.1K.txt', 'gold':'gold.caltech.critical.1K.test'},
            {'txt':'caltech.pedestrians.1K.txt', 'gold':'gold.caltech.1K.test'},
            {'txt':'voc.2012.1K.txt', 'gold':'gold.voc.2012.1K.test'},
]


def main(board):
    print "Generating py-faster for CUDA on " + str(board)

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)

        installDir = config.get('DEFAULT', 'installdir') + "/"
        varDir = config.get('DEFAULT', 'vardir') + "/"
        logDir = config.get('DEFAULT', 'logdir') + "/"
        tmpDir = config.get('DEFAULT', 'tmpdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    data_path = installDir + "data/py_faster_rcnn"
    bin_path = installDir + "bin"
    src_py_faster = installDir + "src/cuda/py-faster-rcnn"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    generate = [str("cd " + src_py_faster),]
    execute = []

    for i in DATASETS:
        gold = data_path + '/' + i['gold']
        txt_list = installDir + 'data/networks_img_list/' + i['txt']
        gen = {
            'gold': [' --gen ', gold],
            'iml': [' --iml ', txt_list],
        'ite': [' --ite ', '1000'], 'zexe': ['sudo ', src_py_faster + "/tools/py_faster_rcnn.py "],
        }

        exe = copy.deepcopy(gen)
        exe['gold'][0] = ' --gld '
        exe['log'] = [' --log ', ' daniel_logs ']

        generate.append(" ".join([''.join(map(str, value)) for key, value in gen.iteritems()]))
        execute.append(" ".join([''.join(map(str, value)) for key, value in exe.iteritems()]))


    #os.system("cd " + src_py_faster)
    for i in generate:
        if os.system(str(i)) != 0:
            print "Something went wrong with generate of " , str(i)
            exit(1)


    fp = open(installDir + "scripts/how_to_run_py_faster_rcnn_cuda_" + board, 'w')

    for i in execute:
        print >> fp, "[\"" + str(i) + "\" , 0.016, \"py_faster_rcnn.py\"],"
        print "[\"" + str(i) + "\" , 0.016, \"py_faster_rcnn.py\"],"

    print "\nConfiguring done, to run check file: " + installDir + "scripts/how_to_run_py_faster_rcnn_cuda_" + str(
        board) + "\n"

    sys.exit(0)


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/k40>"
    else:
        main(str(parameter[0]).upper())


# for caltech
# gold_caltec = gold = data_path + '/gold.caltech.1K.test'
# txt_list_caltec = installDir + 'data/networks_img_list/caltech.pedestrians.1K.txt'
# caltech_gen = {
#     'gold': [' --gen ', gold_caltec],
#     'iml': [' --iml ', txt_list_caltec],
# 'ite': [' --ite ', '1000'], 'zexe': ['sudo ', src_py_faster + "/tools/py_faster_rcnn.py "],
# }
#
# caltech_exe = copy.deepcopy(caltech_gen)
# caltech_exe['gold'][0] = ' --gld '
# caltech_exe['log'] = [' --log ', ' daniel_logs ']
#
# ./py_faster_rcnn.py --gld test.test --iml /home/carol/radiation-benchmarks/data/networks_img_list/caltech/K40/caltech.pedestrians.DEBUG.txt --log daniel_logs
# ./py_faster_rcnn.py --gen test.test --iml /home/carol/radiation-benchmarks/data/networks_img_list/caltech/K40/caltech.pedestrians.DEBUG.txt