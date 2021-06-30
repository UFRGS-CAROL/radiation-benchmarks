#!/usr/bin/python3


import configparser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [1024]
ITERATIONS = 1000000
SIMTIME = [100]
STREAMS = 4
PRECISIONS = ["single"]
BUILDPROFILER = 0

COMPILER_VERSION = [
    ("10.2", "g++"),
    # ("11.3", "g++")
]

COMPILER_FLAGS = (
    # append to parameter list the number of the registers
    # '--maxrregcount=16',
    # '"-Xptxas -O0 -Xcompiler -O0"',
    # '"-Xptxas -O1 -Xcompiler -O1"',
    # # Baseline
    '"-Xptxas -O3 -Xcompiler -O3"',
    # # Fast math implies --ftz=true --prec-div=false --prec-sqrt=false --fmad=true.
    # "--use_fast_math",
)


def config(board, arith_type, debug, compiler_version, flag):
    cuda_version = compiler_version[0]
    cxx_version = compiler_version[1]
    original_hotspot = "hotspot"
    benchmark_bin = "cuda_hotspot_" + arith_type
    benchmark_src = "hotspot"
    print(f"Generating {benchmark_bin} for CUDA, board:{board}")

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = configparser.RawConfigParser()
        config.read(conf_file)
        installDir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise ValueError("Configuration setup error: " + str(e))

    data_path = installDir + "data/" + original_hotspot
    bin_path = installDir + "bin"
    src_hotspot = installDir + "src/cuda/" + benchmark_src

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0o777)
        os.chmod(data_path, 0o777)

    # change it for lava
    generate = ["cd " + src_hotspot,
                "make clean",
                "make -C ../../include ",
                # "make -C ../common/",
                # "make PRECISION={} BUILDPROFILER={}".format(arith_type, BUILDPROFILER),
                "mkdir -p " + data_path,
                # "mv ./" + benchmark_bin + " " + bin_path + "/"
                ]
    execute = []
    flag_parsed = flag.replace("=", "").replace("--", "")
    benchmark_bin_new = f"{benchmark_bin}_{cuda_version}_{flag_parsed}"
    for i in SIZES:
        for s in SIMTIME:
            new_binary = f"{bin_path}/{benchmark_bin_new}"
            cuda_path = f"/usr/local/cuda-{cuda_version}"

            gen = [
                [f'sudo env LD_LIBRARY_PATH={cuda_path}/''lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} '
                 f"{new_binary}"],
                [f'-size={i}'],
                [f'-input_temp={data_path}/temp_{i}'],
                [f'-input_power={data_path}/power_{i}'],  # change for execute
                [f'-gold_temp={data_path}/gold_{i}_{arith_type}_{s}_{cuda_version}_{flag_parsed}'],
                [f'-sim_time={s}'],
                [f'-streams={STREAMS}'],
            ]
            # change mode and iterations for exe
            exe = copy.deepcopy(gen)
            exe.append(['-iterations=' + str(ITERATIONS)])
            gen.extend([['-generate'], ['-iterations=1']])

            variable_gen = ["make clean",
                            f"make LOGS=1 PRECISION={arith_type} NVCCOPTFLAGS={flag}"
                            f" CXX={cxx_version} CUDAPATH={cuda_path}",
                            f"sudo rm -f {new_binary}",
                            f"sudo mv ./{benchmark_bin} {new_binary}"
                            ]

            generate.extend(variable_gen)
            generate.append(' '.join(str(r) for v in gen for r in v))
            execute.append(' '.join(str(r) for v in exe for r in v))

    # execute, generate, install_dir, benchmark_bin, debug
    execute_and_write_json_to_file(execute=execute, generate=generate,
                                   install_dir=installDir,
                                   benchmark_bin=benchmark_bin_new,
                                   debug=debug)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[1:][0]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except IndexError:
        debug_mode = False

    board, _ = discover_board()
    for p in PRECISIONS:
        for compiler_ver in COMPILER_VERSION:
            for compiler_flag in COMPILER_FLAGS:
                config(board=board, arith_type=p, debug=debug_mode, compiler_version=compiler_ver, flag=compiler_flag)
    print("Multiple jsons may have been generated.")
