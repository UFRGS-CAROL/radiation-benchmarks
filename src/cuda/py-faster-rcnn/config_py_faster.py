#!/usr/bin/python

import os
import sys
import ConfigParser
import copy


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

    # ./py_faster_rcnn.py --gld test.test --iml /home/carol/radiation-benchmarks/data/networks_img_list/caltech/K40/caltech.pedestrians.DEBUG.txt --log daniel_logs
    # ./py_faster_rcnn.py --gen test.test --iml /home/carol/radiation-benchmarks/data/networks_img_list/caltech/K40/caltech.pedestrians.DEBUG.txt

    # for voc2012
    gold_voc = gold = data_path + '/gold.voc.2012.1K.test'
    txt_list_voc = installDir + 'data/networks_img_list/voc.2012.1K.txt'
    voc_gen = {
        'exe': ['sudo ', src_py_faster + "/tools/py_faster_rcnn.py "],
        'gold': [' --gen ', gold_voc],
        'iml': [' --iml ', txt_list_voc],
    }

    voc_exe = copy.deepcopy(voc_gen)
    voc_exe['gold'][0] = ' --gld '
    voc_exe['log'] = [' --log ', ' daniel_logs ']

    # for caltech
    gold_caltec = gold = data_path + '/gold.caltech.1K.test'
    txt_list_caltec = installDir + 'data/networks_img_list/caltech.pedestrians.1K.txt'
    caltech_gen = {
        'exe': ['sudo ', src_py_faster + "/tools/py_faster_rcnn.py "],
        'gold': [' --gen ', gold_caltec],
        'iml': [' --iml ', txt_list_caltec],
    }

    caltech_exe = copy.deepcopy(caltech_gen)
    caltech_exe['gold'][0] = ' --gld '
    caltech_exe['log'] = [' --log ', ' daniel_logs ']

    os.system("cd " + src_py_faster)

    generate = []
    generate.append(" ".join([''.join(map(str, value)) for key, value in caltech_gen.iteritems()]))
    generate.append(" ".join([''.join(map(str, value)) for key, value in voc_gen.iteritems()]))

    execute = []
    execute.append(" ".join([''.join(map(str, value)) for key, value in caltech_exe.iteritems()]))
    execute.append(" ".join([''.join(map(str, value)) for key, value in voc_exe.iteritems()]))

    for i in generate:
        os.system(str(i))
        print i
        break
        #print i

    fp = open(installDir + "scripts/how_to_run_py_faster_rcnn_cuda_" + board, 'w')

    for i in execute:
        print >> fp, "[\"""sudo " + str(i) + "\" , 0.016, \"py_faster_rcnn\"],"
        #print "[\"""sudo " + str(i) + "\" , 0.016, \"py_faster_rcnn\"],"

    print "\nConfiguring done, to run check file: " + installDir + "scripts/how_to_run_py_faster_rcnn_cuda_" + str(
        board) + "\n"

    sys.exit(0)


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/k40>"
    else:
        main(str(parameter[0]).upper())
