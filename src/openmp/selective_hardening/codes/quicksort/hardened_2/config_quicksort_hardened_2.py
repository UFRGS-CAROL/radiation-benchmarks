#!/usr/bin/python

import ConfigParser
import sys

sys.path.insert(0, '../../../../../include')
from common_config import discover_board, execute_and_write_json_to_file

NUM_THREADS = 4
INPUT_SIZE = 2097152
INPUT_FILE = "inputsort_" + str(INPUT_SIZE)
GOLD_FILE = "gold_" + str(INPUT_SIZE)
ITERATIONS = 100000

def config(board, debug):

    benchmark_bin = "quick_check_hardened_2"
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
    src_benchmark = install_dir + "src/openmp/selective_hardening/codes/quicksort/hardened_2"
    selective_hardening_dir = "/var/selective_hardening/"

    generate = ["sudo mkdir -p " + bin_path,
                "sudo mkdir -p " + selective_hardening_dir,
                "cd " + src_benchmark, 
                "make clean",
                "make",
                "./genInput {}".format(str(INPUT_SIZE)),
                "./quick_gen {} {} {}".format(str(INPUT_SIZE), str(NUM_THREADS), INPUT_FILE),
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/",
                "make clean"]
    
    execute = []

    exe = [None] * 6
    exe[0] = [bin_path + "/" + benchmark_bin]
    exe[1] = [str(INPUT_SIZE)]
    exe[2] = [str(NUM_THREADS)]
    exe[3] = [INPUT_FILE]
    exe[4] = [GOLD_FILE]
    exe[5] = [str(ITERATIONS)]

    execute = [(' '.join(str(r) for v in exe for r in v))]
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
