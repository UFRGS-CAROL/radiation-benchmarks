#!/usr/bin/python

import ConfigParser
import copy
import os
import sys
import shutil

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

TYPES = ["L1", "SHARED", "REGISTERS"] #, "CONSTANT"]
ITERATIONS = 100000
SLEEPONGPU = 1


def config(board, debug):

    benchmark_bin = "cudaCacheTest"
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
    src_benchmark = install_dir + "src/cuda/cache"

    generate = ["sudo mkdir -p " + bin_path,
                "cd " + src_benchmark, 
                "make clean", 
                "make -C ../../include ", 
                "make DISABLEL1CACHE=0 -j 4 LOGS=1",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/",
                "make clean",
                "make DISABLEL1CACHE=1 -j 4 LOGS=1",
                "sudo mv -f ./" + benchmark_bin + "L2 " + bin_path + "/"]
    execute = []

    # gen only for max size, defined on cuda_trip_mxm.cu
    for i in TYPES:
        exe = [None] * 5
        exe[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ', bin_path + "/" + benchmark_bin + " "]
        exe[1] = ['--iterations {}'.format(ITERATIONS)]
        exe[2] = ['--sleepongpu {}'.format(SLEEPONGPU)]
        exe[3] = ['--memtotest ' + i]
        exe[4] = ['--verbose 1']

        execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)  

 
    benchmark_bin += "L2"
    exe = [None] * 5
    exe[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ', bin_path + "/" + benchmark_bin + " "]
    exe[1] = ['--iterations {}'.format(ITERATIONS)]
    exe[2] = ['--sleepongpu {}'.format(SLEEPONGPU)]
    exe[3] = ['--memtotest ' + i]
    exe[4] = ['--verbose 1']

    execute = [(' '.join(str(r) for v in exe for r in v))]
    execute_and_write_json_to_file(execute, [], install_dir, benchmark_bin, debug=debug)  


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[1:][1]).upper() 
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False
    
    board, _ = discover_board()
    config(board=board, debug=debug_mode)
    print "A json has been generated."
