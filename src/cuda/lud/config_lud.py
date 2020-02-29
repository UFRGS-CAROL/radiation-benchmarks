#!/usr/bin/python3
import configparser
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

ITERATIONS = 2147483647
SIZES = [4096]
BUILDPROFILER = 1


def config(board, debug):
    benchmark_bin = "cudaLUD"
    print("Generating {} for CUDA, board: {}".format(benchmark_bin, board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = configparser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise ValueError("Configuration setup error: " + str(e))

    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/lud"
    data_path = install_dir + "data/accl"

    generate = ["sudo mkdir -p " + bin_path,
                "sudo mkdir -p " + data_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include ",
                "make -C ../common ",
                "make BUILDPROFILER={} LOGS=1".format(BUILDPROFILER),
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]

    execute = []
    for size in SIZES:
        gold_path = data_path + "/gold_{}.data".format(size)
        input_path = data_path + "/input_{}.data".format(size)
        # ./ --size 8192 --generate --input ./input_8192.data --gold ./gold_8192.data --iterations 10 --verbose;

        gen = [None] * 6
        gen[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${'
                  'LD_LIBRARY_PATH}} ',
                  bin_path + '/' + benchmark_bin + " "]
        gen[1] = ['--iterations {}'.format(ITERATIONS)]
        gen[2] = ['--size {}'.format(size)]
        gen[3] = ['--input {}'.format(input_path)]
        gen[4] = ['--gold {}'.format(gold_path)]
        gen[5] = ['--generate ']

        command = ' '.join(str(r) for v in gen for r in v)
        generate.append(command)
        del gen[-1]
        command = ' '.join(str(r) for v in gen for r in v)
        execute.append(command)

    if debug is False:
        execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)
    else:
        for g in generate:
            print(g)
        print()
        for e in execute:
            print(e)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[1:][0]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        pass

    board, _ = discover_board()
    config(board=board, debug=debug_mode)
