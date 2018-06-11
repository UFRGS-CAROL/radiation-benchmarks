#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

ITERATIONS = 10000


def config(board, arith_type, debug):

    benchmark_bin = "cuda_trip_micro-fma_" + arith_type
    print "Generating " + benchmark_bin + " for CUDA, board:" + board

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/trip_micro-fma"

    generate = ["sudo mkdir -p " + bin_path, 
                "cd " + src_benchmark, 
                "make clean", 
                "make -C ../../include ", 
                "make TYPE=" + arith_type + " -j 4",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    exe = [None] * 2
    exe[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ', bin_path + '/' + benchmark_bin + " "]
    exe[1] = ['-iterations=' + str(ITERATIONS)]

    execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)



if __name__ == "__main__":
    try:
        parameter = str(sys.argv[1:][0])
        if parameter == 'TYPE=double':
            arith_type = 'double'
        if parameter == 'TYPE=single':
            arith_type = 'single'
        if parameter == 'TYPE=half':
            arith_type = 'half'
    except:
        print "Usage: config_trip_micro-fma.py TYPE=<double|single|half> [DEBUG]"
        exit()

    try:
        parameter = str(sys.argv[1:][1]).upper() 
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False
    
    board, _ = discover_board()
    config(board=board, arith_type=arith_type, debug=debug_mode)
