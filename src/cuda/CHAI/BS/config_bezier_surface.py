#!/usr/bin/python


import copy
import os
import sys
import ConfigParser

INPUT = ['input/control.txt']
ITERATIONS = 100000
ALPHA_VARIATIONS = [1.0, 0.0, 0.1]
RESOLUTIONS = [2500, 5000, 10000]

EMBEDDED_HOSTS = ['K1', 'X1', 'X2', 'APU']

DEBUG_MODE = True


def main(board):
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

    generate = ["cd " + src_bs, "make clean", "make -j4",
                "mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for i in INPUT:
        for j in ALPHA_VARIATIONS:
            if j > 0 and board not in EMBEDDED_HOSTS:
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
         "make -j4 LOGS=1",
         "mv -f ./" + benchmark_bin + " " + bin_path + "/"])
    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin)


def execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin):
    for i in generate:
        print i
        if not DEBUG_MODE:
            if os.system(str(i)) != 0:
                print "Something went wrong with generate of ", str(i)
                exit(1)

    list_to_print = ["[\n"]
    for ii, i in enumerate(execute):
        command = "{\"killcmd\": \"killall -9 " + benchmark_bin + "\", \"exec\": \"" + str(i) + "\"}"
        if ii != len(execute) - 1:
            command += ',\n'
        list_to_print.append(command)
    list_to_print.append("\n]")

    with open(install_dir + "scripts/json_files/" + benchmark_bin + ".json", 'w') as fp:
        fp.writelines(list_to_print)

    print "\nConfiguring done, to run check file: " + install_dir + "scripts/json_files/" + benchmark_bin + ".json"


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/x2/k40/titan>"
    else:
        main(str(parameter[0]).upper())
