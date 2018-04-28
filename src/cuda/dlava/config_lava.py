#!/usr/bin/python


import ConfigParser
import copy
import os
import sys

BOXES=[10, 15, 20, 25]
ITERATIONS=10000

def main(board):
    print "Generating Lava for CUDA, board:" + board

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)
        installDir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    benchmark_bin = "lava"
    data_path = installDir + "data/lava"
    bin_path = installDir + "bin"
    src_lava = installDir + "src/cuda/lavaMD"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)


    # change it for lava
    generate = ["cd " + src_lava, "make clean", "make -C ../../include ", "make", "mkdir -p " + data_path,
                "mv ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for i in BOXES:
        inputFile = data_path + "/"

        #	$(RAD_BENCH)/src/cuda/bfs/$(EXE) -t 0 -f $(RAD_BENCH)/data/bfs/graph1MW_6.txt -c temp.gold -m 1 -r 100
        gen = [None] * 8
        gen[0] = ['sudo ', bin_path + "/" + benchmark_bin + " "]
        gen[1] = ['-boxes=' + str(i)]
        gen[2] = ['-generate ']
        gen[3] = ['-output_gold=' + inputFile + "gold_" +  str(i) + ""]
        gen[4] = ['-iterations=1']  # change for execute
        gen[5] = ['-streams=1']
        gen[6] = ['-input_charges=' + inputFile + "input_charges_" + str(i)]
        gen[7] = ['-input_distances=' + inputFile + "input_distances_" + str(i)]

        # change mode and iterations for exe
        exe = copy.deepcopy(gen)
        exe[2] = []
        exe[4] = ['-iterations=' + str(ITERATIONS)]

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
