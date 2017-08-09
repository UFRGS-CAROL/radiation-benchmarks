#!/usr/bin/python
import os
import sys
import ConfigParser
import sys
import copy

DATASETS = [
    # normal
    {'txt': 'caltech.pedestrians.1K.txt', 'gold': 'gold.caltech.1K.csv'},
    {'txt': 'urban.street.1.1K.txt', 'gold': 'gold.urban.street.1.1K.csv'},
    {'txt': 'voc.2012.1K.txt', 'gold': 'gold.voc.2012.abft.1K.csv'},
]

def download_weights(src_dir, data_dir):
    os.chdir(data_dir)
    if os.system(".get_darknet_weights.sh") != 0:
        print "ERROR on downloading darknet v1/v2 weights"
        exit(-1)

    os.chdir(src_dir)


def main(board):
    print "Generating darknet for CUDA, board:" + board

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

    data_path = installDir + "data/darknet"
    bin_path = installDir + "bin"
    src_darknet = installDir + "src/cuda/darknet_v2"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    # executing weights test first
    download_weights(src_dir=src_darknet, data_dir=data_path)

    #change it for darknetv2
    generate = ["cd " + src_darknet, "make clean GPU=1", "make -j4 GPU=1 ", "mv ./darknet_v2 " + bin_path + "/"]
    execute = []
    for i in DATASETS:
        abft = 0
        # ./ $(EXEC)
        # test_radiation - c $(RAD_DIR) / src / cuda / darknet_v2 / cfg / yolo.cfg \
        #                     - w $(RAD_DIR) / data / darknet / yolo_v2.weights \
        #                          - g
        # 1 - d $(RAD_DIR) / data / darknet / fault_injection.csv - s
        # 1 - l \
        #         $(RAD_DIR) / data / networks_img_list / fault_injection.txt - a 0
        gold = data_path + '/' + i['gold']
        txt_list = installDir + 'data/networks_img_list/' + i['txt']
        gen = {
            'bin': [bin_path, "/darknet_v2"],
            # 'e': [' -e ', 'yolo'],  # execution_type =
            'aa': ['test_radiation', ''],  # execution_model =
            'c': [' -c ', data_path + '/yolo_v2.cfg'],  # config_file =
            'w': [' -w ', data_path + '/yolo_v2.weights'],  # weights =
            'n': [' -n ', '1'],  # iterations =  #it is not so much, since each dataset have at least 10k of images
            'g': [' -g ', gold],  # base_caltech_out = base_voc_out = src_darknet
            'l': [' -l ', txt_list],
            # 'b': [' -b ', src_darknet],
            # 'x': [' -x ', 0],
            's': [' -s ', 0],
            'a': [' -a ', abft],
        }

        exe = copy.deepcopy(gen)
        exe['n'][1] = 10000
        exe['g'][0] = ' -d '

        exe_save = copy.deepcopy(exe)
        exe_save['s'][1] = 1


        generate.append(" ".join([''.join(map(str, gen[key])) for key in gen]))

        execute.append(" ".join([''.join(map(str, value)) for key, value in exe.iteritems()]))
        execute.append(" ".join([''.join(map(str, value)) for key, value in exe_save.iteritems()]))



    # end for generate

    generate.append("make clean GPU=1 ")
    generate.append("make -C ../../include/")
    generate.append("make -j 4 GPU=1 LOGS=1")
    generate.append("mv ./darknet_v2 " + bin_path + "/")

    for i in generate:
        if os.system(str(i)) != 0:
            print "Something went wrong with generate of ", str(i)
            exit(1)
        # print i, "\n"

    fp = open(installDir + "scripts/how_to_run_darknet_v2_cuda_" + board, 'w')

    for i in execute:
        print >> fp, "[\"""sudo " + str(i) + "\" , 0.016, \"darknet_v2\"],"

    print "\nConfiguring done, to run check file: " + installDir + "scripts/how_to_run_darknet_v2_cuda_" + board + "\n"

    sys.exit(0)


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/x2/k40/titan>"
    else:
        main(str(parameter[0]).upper())