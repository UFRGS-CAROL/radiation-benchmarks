#!/usr/bin/python


import copy
import os
import sys
import ConfigParser

sys.path.insert(0, '../../../include')
from common_config import discover_board, execute_and_write_json_to_file

INPUT = ['input/control.txt']
ITERATIONS = 100000
ALPHA_VARIATIONS = [1.0, 0.0, 0.1]
RESOLUTIONS = [2500]

EMBEDDED_HOSTS = ['K1', 'TX1', 'TX2', 'CarolTegraX1A']

def config(board, debug):
    print "Generating Bezier Surface for CUDA on " + str(board)

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)

        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    benchmark_bin = "bezier_surface"
    data_path = install_dir + "data/" + benchmark_bin
    bin_path = install_dir + "bin"
    src_bs = install_dir + "src/cuda/CHAI/BS"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    generate = ["mkdir -p " + bin_path, "cd " + src_bs, "make clean", "make ",
                "mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for i in INPUT:
        for j in ALPHA_VARIATIONS:
            if j > 0.0 and board not in EMBEDDED_HOSTS:
                continue
            for r in RESOLUTIONS:
                if r > 2500 and board in EMBEDDED_HOSTS:
                    continue
                input_file = data_path + "/" + i

                # ./ gen_gold - f / home / carol / radiation - benchmarks / data / bezier_surface / input / control.txt - n
                # 2500 \
                # - d / home / carol / radiation - benchmarks / data / bezier_surface / input.bin \
                # - g / home / carol / radiation - benchmarks / data / bezier_surface / bezier_surface_2500.gold
                gen = [None] * 8
                gen[0] = ['sudo ', src_bs + "/gen_gold "]
                gen[1] = ['-d ', data_path + "/input.bin"]
                gen[2] = ['-r ', 1]
                gen[3] = ['-a ', j]
                gen[4] = ['-s ', 0]  # change for execute
                gen[5] = ['-z ',
                          data_path + "/alpha_" + str(j) + "_in_size_" + str(r) + "_out_size_" + str(r) + ".gold"]
                gen[6] = ['-f ', input_file]
                gen[7] = ['-n ', r]

                # change mode and iterations for exe
                exe = copy.deepcopy(gen)
                exe[0][1] = bin_path + "/" + benchmark_bin + " "
                exe[4][1] = 1
                exe[2][1] = ITERATIONS

                generate.append(' '.join(str(r) for v in gen for r in v))
                execute.append(' '.join(str(r) for v in exe for r in v))

    generate.extend(
        ["make clean", "make -C ../../../include/",
         "make  LOGS=1",
         "mv -f ./" + benchmark_bin + " " + bin_path + "/"])


    #execute, generate, install_dir, benchmark_bin, debug
    execute_and_write_json_to_file(execute=execute, generate=generate,
                                    install_dir=install_dir, 
                                    benchmark_bin=benchmark_bin, 
                                    debug=debug)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[1:][0]).upper() 
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False
    
    board, _ = discover_board()
    config(board=board, debug=debug_mode)
    
