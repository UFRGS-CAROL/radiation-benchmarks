#!/usr/bin/python


import copy
import os
import sys
import ConfigParser

INPUT = ['input/control.txt']
ITERATIONS = 100000
ALPHA_VARIATIONS = [0, 0.1, 0.2, 0.3]
RESOLUTIONS = [2500, 5000]


def main(board):
    print "Generating Bezier Surface for CUDA on " + str(board)

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)

        installDir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    benchmark_bin = "bezier_surface"
    data_path = installDir + "data/" + benchmark_bin
    bin_path = installDir + "bin"
    src_srad = installDir + "src/cuda/" + benchmark_bin

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    generate = ["cd " + src_srad, "make clean", "make -j4",
                "mv ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for i in INPUT:
        for j in ALPHA_VARIATIONS:
            if j > 0 and ('X1' not in board and 'X2' not in board and 'K1' not in board):
                continue
            for r in RESOLUTIONS:
                if r > 2500 and ('X1' in board or 'X2' in board or 'K1' in board):
                    continue
                inputFile = data_path + "/" + i

                # $(RAD_BENCH) / src / cuda / bezier_surface /$(EXE) - w
                # 0 - r
                # 10 - a
                # 0 - s
                # 1 \
                # - z $(RAD_BENCH) / data / bezier_surface / temp.gold \
                #      - f $(RAD_BENCH) / data / bezier_surface / input / control.txt - n
                # 2500
                gen = [None] * 8
                gen[0] = ['sudo ', bin_path + "/" + benchmark_bin + " "]
                gen[1] = ['-w ', 0]
                gen[2] = ['-r ', 1]
                gen[3] = ['-a ', j]
                gen[4] = ['-s ', 0]  # change for execute
                gen[5] = ['-z ',
                          data_path + "/alpha_" + str(j) + "_in_size_" + str(r) + "_out_size_" + str(r) + ".gold"]
                gen[6] = ['-f ', inputFile]
                gen[7] = ['-n ', r]

                # change mode and iterations for exe
                exe = copy.deepcopy(gen)
                exe[4][1] = 1
                exe[2][1] = ITERATIONS

                generate.append(' '.join(str(r) for v in gen for r in v))
                execute.append(' '.join(str(r) for v in exe for r in v))

    generate.extend(
        ["make clean", "make -C ../../include/",
         "make -j4 LOGS=1",
         "mv ./" + benchmark_bin + " " + bin_path + "/"])
    execute_and_write_how_to_file(execute, generate, installDir, benchmark_bin)


def execute_and_write_how_to_file(execute, generate, installDir, benchmark_bin):
    for i in generate:
        if os.system(str(i)) != 0:
            print "Something went wrong with generate of ", str(i )
            exit(1)
        print i
    fp = open(installDir + "scripts/json_files/" + benchmark_bin + ".json", 'w')

    list_to_print = ["["]
    for ii, i in enumerate(execute):
        command = "{\"killcmd\": \"killall -9 " + benchmark_bin + "\", \"exec\": \"" + str(i) + "\"}"
        if ii != len(execute) - 1:
            command += ', '
        list_to_print.append(command)
    list_to_print.append("]")

    for i in list_to_print:
        print >> fp, i
        print i
    fp.close()
    print "\nConfiguring done, to run check file: " + installDir + "scripts/json_files/" + benchmark_bin + ".json"


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/k40>"
    else:
        main(str(parameter[0]).upper())
