#!/usr/bin/python


import ConfigParser
import copy
import os

import sys

SIZES=[1024]
ITERATIONS=10000
SIMTIME=[1000]

def main(board):

    benchmark_bin = "hotspot"
    print "Generating "+ benchmark_bin + " for CUDA, board:" + board

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)
        installDir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)


    data_path = installDir + "data/" + benchmark_bin
    bin_path = installDir + "bin"
    src_hotspot = installDir + "src/cuda/" + benchmark_bin

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)


    # change it for lava
    generate = ["cd " + src_hotspot, "make clean", "make -C ../../include ", "make", "mkdir -p " + data_path,
                "mv ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for i in SIZES:
        for s in SIMTIME:
            inputFile = data_path + "/"

            gen = [None] * 8
            gen[0] = ['sudo ', bin_path + "/" + benchmark_bin + " "]
            gen[1] = ['-size=' + str(i)]
            gen[2] = ['-generate ']
            gen[3] = ['-temp_file=' + inputFile + "temp_" +  str(i)]
            gen[4] = ['-power_file=' + inputFile + "power_" + str(i)]  # change for execute
            gen[5] = ['-gold_file=' + inputFile + "gold_" + str(i) + "_" + str(s)]
            gen[6] = ['-sim_time=' + str(i)]
            gen[7] = ['-iterations=1']

            # change mode and iterations for exe
            exe = copy.deepcopy(gen)
            exe[2] = []
            exe[7] = ['-iterations=' + str(ITERATIONS)]

            generate.append(' '.join(str(r) for v in gen for r in v))
            execute.append(' '.join(str(r) for v in exe for r in v))


    execute_and_write_how_to_file(execute, generate, installDir, benchmark_bin)

def execute_and_write_how_to_file(execute, generate, installDir, benchmark_bin):
    for i in generate:
        if os.system(str(i)) != 0:
            print "Something went wrong with generate of ", str(i)
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
        print "./config_generic <k1/x1/x2/k40/titan>"
    else:
        main(str(parameter[0]).upper())
