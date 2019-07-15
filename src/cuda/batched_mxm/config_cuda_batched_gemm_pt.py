#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [128]
STREAMS = 1024
KERNELTYPE=[0, 1, 2] # STATIC, PERSISTENT, GEMM
ITERATIONS = int(1e9)


def config(board, debug):

    DATA_PATH_BASE = "mxm_single"

    benchmark_bin = "cuda_batched_mxm"
    print("Generating {} for CUDA, board: {}".format(benchmark_bin, board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    data_path = install_dir + "data/" + DATA_PATH_BASE
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/batched_mxm"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    for_jetson = 0
    lib = "NVMLWrapper.so"
    if "X1" in board or "X2" in board:
        for_jetson = 1
        lib = ""

    generate = ["sudo mkdir -p " + bin_path, 
                "cd " + src_benchmark, 
                "make clean", 
                "make -C ../../include ",
                "make -C ../common {}".format(lib),
                "make FORJETSON={} -j2".format(for_jetson),
                "mkdir -p " + data_path,
                "sudo rm -f " + data_path + "/*" + benchmark_bin + "*",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    # gen only for max size, defined on cuda_trip_mxm.cu
    for i in SIZES:
        for k in KERNELTYPE:
            input_file = data_path + "/"

            gen = [None] * 8
            gen[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ',
                      bin_path + "/" + benchmark_bin + " "]
            gen[1] = ['-size={}'.format(i)]
            gen[2] = ['-input_a={}A_pt_streams_{}_ktype_{}_size_{}.matrix'.format(input_file, STREAMS, k, i)]
            gen[3] = ['-input_b={}B_pt_streams_{}_ktype_{}_size_{}.matrix'.format(input_file, STREAMS, k, i)]
            gen[4] = ['-gold={}GOLD_pt_streams_{}_ktype_{}_size_{}.matrix'.format(input_file, STREAMS, k, i)]
            gen[5] = ['-generate']
            gen[6] = ['-kernel_type={}'.format(k)]
            gen[7] = ['-batch={}'.format(STREAMS)]

            # change mode and iterations for exe
            exe = copy.deepcopy(gen)
            exe[5] = ['-iterations={}'.format(ITERATIONS)]

            generate.append(' '.join(str(r) for v in gen for r in v))
            execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)



if __name__ == "__main__":
    debug_mode = None
    try:
        parameter = str(sys.argv[1:][1]).upper() 
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False
    
    board, hostname = discover_board()
    if hostname is None:
        hostname = "carolgeneric"
    config(board=hostname, debug=debug_mode)
    print("Multiple jsons may have been generated.")
