#!/usr/bin/python


import copy
import os
import sys
import ConfigParser

INPUT = ["lakes_graph_in"]
ITERATIONS = 100000

#  ./bfs -f input/lakes_graph_in -c output/lakes_graph_out -r 1 -l 900000000  #CPU
# ./bfs -f input/lakes_graph_in -c output/lakes_graph_out -r 1 -l 0  #GPU
# ./bfs -f input/lakes_graph_in -c output/lakes_graph_out -r 1 -l 128 #CPU+GPU

THREADS_HOST = [900000000, 0, 128]
EMBEDDED_HOSTS = ['K1', 'X1', 'X2', 'APU']

DEBUG_MODE = False


def untar_graphs(file_path):
    tries = 3

    while not os.path.isfile(file_path + "/lakes_graph_in"):
        print "tar xzf " + file_path + "/lakes_graph_in.tar.gz -C " + file_path + "/"
        if os.system("tar xzf " + file_path + "/lakes_graph_in.tar.gz -C " + file_path + "/") != 0:
            print "Something went wrong with untar of " + file_path + " file. Trying again"

        if tries == 0:
            return False
        tries -= 1

    return True


def main(board):
    print "Generating BFS for CUDA on " + str(board)

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)

        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    benchmark_bin = "bfs"
    data_path = install_dir + "data/" + benchmark_bin
    bin_path = install_dir + "bin"
    src_bfs = install_dir + "src/cuda/CHAI"

    if not untar_graphs(data_path):
        raise ValueError("Error on untar the file")

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    generate = ["mkdir -p " + bin_path, "cd " + src_bfs, "make clean", "make -j4",
                "mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for i in INPUT:
        for j in THREADS_HOST:
            if j > 0 and board not in EMBEDDED_HOSTS:
                continue

            input_file = data_path + "/" + i
            output_file = data_path + "/lakes_graph_out"
            # $(RAD_BENCH)/src/cuda/bfs/$(EXE) -t 0 -f $(RAD_BENCH)/data/bfs/graph1MW_6.txt -c temp.gold -m 1 -r 100
            gen = [None] * 7
            gen[0] = ['sudo ', bin_path + "/" + benchmark_bin + " "]
            gen[1] = ['-l ', j]
            gen[2] = ['-f ', input_file]
            gen[3] = ['-p ', input_file + ".gold"]
            gen[4] = ['-m ', 0]  # change for execute
            gen[5] = ['-r ', 1]
            gen[6] = ['-c ', output_file]

            # change mode and iterations for exe
            exe = copy.deepcopy(gen)
            exe[4][1] = 1
            exe[5][1] = ITERATIONS

            generate.append(' '.join(str(r) for v in gen for r in v))
            execute.append(' '.join(str(r) for v in exe for r in v))

    generate.extend(
        ["make clean", "make -C ../../../include/",
         "make LOGS=1",
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
    global DEBUG_MODE

    parameter = sys.argv[1:]
    try:
        DEBUG_MODE = sys.argv[2:]
    except:
        DEBUG_MODE = False
    if len(parameter) < 1:
        print "./config_generic <k1/x1/x2/k40/titan>"
    else:
        main(str(parameter[0]).upper())
