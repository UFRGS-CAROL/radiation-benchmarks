#!/usr/bin/python

import ConfigParser
import sys

sys.path.insert(0, '/home/carol/radiation-benchmarks/src/include')
from common_config import discover_board, execute_and_write_json_to_file

ITERATIONS = 0 # zero means the max number possible
NUM_THREADS = 8 # 8 threads for Ryzen, 4 Threads for Steamroller
def config(board, debug):

    benchmark_bin = "vmul_double"
    print("Generating {} for OpenMP, board:{}".format(benchmark_bin, board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        sys.stderr.write("Configuration setup error: " + str(e))
        sys.exit(1)

    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/microbenchmarks/vector/mul_double/"

    generate = ["sudo mkdir -p " + bin_path,
                "cd " + src_benchmark, 
                "make clean",
                "make",
                "sudo mv -f "+src_benchmark + benchmark_bin + " " + bin_path + "/",
                "make clean"]

    execute = []
    exe = [None] * 3
    exe[0] = [bin_path + "/" + benchmark_bin]
    exe[1] = [str(ITERATIONS)]
    exe[2] = [str(NUM_THREADS)]
    execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)  



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
    print("A json has been generated.")
