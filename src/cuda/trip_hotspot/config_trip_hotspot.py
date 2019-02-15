#!/usr/bin/python


import ConfigParser
import copy
import os
import sys
sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file


SIZES=[1024]
ITERATIONS=10000
STREAMS=10
PRECISIONS = {"double": 1024, "single": 2048, "half": 4096}

def config(board, arith_type, debug):
    
    original_hotspot = "hotspot"
    benchmark_bin = "cuda_trip_hotspot_"+arith_type
    benchmark_src = "trip_hotspot"
    print "Generating "+ benchmark_bin + " for CUDA, board:" + board

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)
        installDir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)


    data_path = installDir + "data/" + original_hotspot
    bin_path = installDir + "bin"
    src_hotspot = installDir + "src/cuda/" + benchmark_src

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)


    # change it for lava
    generate = ["cd " + src_hotspot, 
                "make clean", 
                "make -C ../../include ", 
                "make -C ../../include/safe_memory/", 
                "make PRECISION=" + arith_type, 
                "mkdir -p " + data_path,
                "mv ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []
    s = PRECISIONS[arith_type]
    for i in SIZES:
        inputFile = data_path + "/"

        gen = [None] * 9
        gen[0] = [' ', bin_path + "/" + benchmark_bin + " "]
        gen[1] = ['-size=' + str(i)]
        gen[2] = ['-generate ']
        gen[3] = ['-input_temp=' + inputFile + "temp_" +  str(i)]
        gen[4] = ['-input_power=' + inputFile + "power_" + str(i)]  # change for execute
        gen[5] = ['-gold=' + inputFile + "gold_" + str(i) + "_" + arith_type + "_"  + str(s)]
        gen[6] = ['-sim_time=' + str(s)]
        gen[7] = ['-iterations=1']
        gen[8] = ['-streams=' + str(STREAMS)]

        # change mode and iterations for exe
        exe = copy.deepcopy(gen)
        exe[2] = []
        exe[7] = ['-iterations=' + str(ITERATIONS)]

        generate.append(' '.join(str(r) for v in gen for r in v))
        execute.append(' '.join(str(r) for v in exe for r in v))


    #execute, generate, install_dir, benchmark_bin, debug
    execute_and_write_json_to_file(execute=execute, generate=generate,
                                    install_dir=installDir, 
                                    benchmark_bin=benchmark_bin, 
                                    debug=debug)




if __name__ == "__main__":
    try:
        parameter = str(sys.argv[1:][0]).upper() 
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False
    
    board, _ = discover_board()
    for p in PRECISIONS:
        config(board=board, arith_type=p, debug=debug_mode)
    print "Multiple jsons may have been generated."
