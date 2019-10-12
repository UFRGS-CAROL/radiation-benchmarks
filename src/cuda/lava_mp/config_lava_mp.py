#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

# Size and streams
SIZES = [[23, 2]]
REDUNDANCY = ["dmrmixed"]
PRECISIONS = ["double"]
ITERATIONS = int(1e9)
DATA_PATH_BASE = "lava"
CHECK_BLOCK = [1, 12]
BUILDPROFILER = 1


def config(board, debug):
    benchmark_bin = "cuda_lava"
    print("Generating " + benchmark_bin + " for CUDA, board:" + board)

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise IOError("Configuration setup error: " + str(e))

    data_path = install_dir + "data/" + DATA_PATH_BASE
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/lava_mp"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 777)
        os.chmod(data_path, 777)

    generate = ["sudo mkdir -p " + bin_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include ",
                "make -j 3 BUILDPROFILER={}".format(BUILDPROFILER),
                "sudo rm -f " + data_path + "/{}*".format(DATA_PATH_BASE),
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []
    for size in SIZES:
        for arith_type in PRECISIONS:
            input_file = data_path + "/"

            gen = [None] * 11
            gen[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ',
                      bin_path + "/" + benchmark_bin + " "]
            gen[1] = ['-boxes {}'.format(size[0])]
            gen[2] = ['-streams {}'.format(size[1])]
            gen[3] = ['-input_distances {}'.format(input_file + 'lava_distances_' + arith_type + '_' + str(size[0]))]
            gen[4] = ['-input_charges {}'.format(input_file + 'lava_charges_' + arith_type + '_' + str(size[0]))]
            gen[5] = ['-output_gold {}'.format(input_file + "lava_gold_" + arith_type + '_' + str(size[0]))]
            gen[6] = ['-iterations {}'.format(ITERATIONS)]
            gen[7] = ['-redundancy none']
            gen[8] = ['-precision {}'.format(arith_type)]
            gen[9] = ['-verbose']
            gen[10] = ['-generate']
            generate.append(' '.join(str(r) for v in gen for r in v))

            for redundancy in REDUNDANCY:
                if redundancy is 'none':
                    # change mode and iterations for exe
                    exe = copy.deepcopy(gen)
                    exe[7] = ['-redundancy {}'.format(redundancy)]
                    exe[9] = ['-opnum 0']
                    exe.pop()
                    execute.append(' '.join(str(r) for v in exe for r in v))
                else:
                    for check in CHECK_BLOCK:
                        # change mode and iterations for exe
                        exe = copy.deepcopy(gen)
                        exe[7] = ['-redundancy {}'.format(redundancy)]
                        exe[9] = ['-opnum {}'.format(check)]
                        exe.pop()

                        execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[0:][1]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False

    board, _ = discover_board()
    config(board=board, debug=debug_mode)
    print("Multiple jsons may have been generated.")
