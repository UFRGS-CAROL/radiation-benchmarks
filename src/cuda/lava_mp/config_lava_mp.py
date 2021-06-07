#!/usr/bin/python3

import configparser
import copy
import os
import re
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

# Size and streams
SIZES = [[16, 2]]
REDUNDANCY = ["none"]
PRECISIONS = ["float"]
ITERATIONS = int(1e9)
DATA_PATH_BASE = "lava"
CHECK_BLOCK = []
BUILDPROFILER = 1

COMPILER_VERSION = [
    ("10.2", "g++"),
    ("11.3", "g++")
]

COMPILER_FLAGS = (
    # append to parameter list the number of the registers
    '--maxrregcount=16',
    '"-Xptxas -O0 -Xcompiler -O0"',
    '"-Xptxas -O1 -Xcompiler -O1"',
    # Baseline
    '"-Xptxas -O3 -Xcompiler -O3"',
    # Fast math implies --ftz=true --prec-div=false --prec-sqrt=false --fmad=true.
    "--use_fast_math",
)


def config(device, compiler, flag, debug):
    benchmark_bin = "cuda_lava"
    cuda_version = compiler[0]
    cxx_version = compiler[1]
    flags_parsed = re.sub("-*=*[ ]*\"*", "", flag)

    new_bench_bin = f"{benchmark_bin}_{cuda_version}_{flags_parsed}"

    print(f"Generating {benchmark_bin} for CUDA")

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = configparser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise IOError("Configuration setup error: " + str(e))

    data_path = install_dir + "data/" + DATA_PATH_BASE
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/lava_mp"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 777)
        os.chmod(data_path, 777)

    generate = [
        "sudo mkdir -p " + bin_path,
        "cd " + src_benchmark,
        "make -C ../../include ",
        f"sudo rm -f {data_path}/*{cuda_version}*{flags_parsed}*",
    ]
    execute = []
    for size in SIZES:
        for arith_type in PRECISIONS:
            new_binary = f"{bin_path}/{new_bench_bin}"
            cuda_path = f"/usr/local/cuda-{cuda_version}"

            input_file = data_path
            input_distances = f"{input_file}/lava_distances_{arith_type}_{size[0]}_{cuda_version}_{flags_parsed}"
            input_charges = f"{input_file}/lava_charges_{arith_type}_{size[0]}_{cuda_version}_{flags_parsed}"
            output_gold = f"{input_file}/lava_gold_{arith_type}_{size[0]}_{cuda_version}_{flags_parsed}"
            execute_cmd = f'sudo env LD_LIBRARY_PATH={cuda_path}/' + 'lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} '
            gen = [
                [execute_cmd, new_binary],
                [f'-boxes {size[0]}'],
                [f'-streams {size[1]}'],
                [f'-input_distances {input_distances}'],
                [f'-input_charges {input_charges}'],
                [f'-output_gold {output_gold}'],
                [f'-iterations {ITERATIONS}'],
                ['-redundancy none'],
                [f'-precision {arith_type}'],
                [f'-redundancy none'], ['-opnum 0']
            ]
            # change mode and iterations for exe
            exe = copy.deepcopy(gen)
            gen.append(['-generate'])
            # gen.append(['--check_input_existence'])
            gen.append(['-verbose'])
            variable_gen = ["make clean",
                            f"make -j 4 LOGS=1 NVCCOPTFLAGS={flag} CXX={cxx_version} CUDAPATH={cuda_path}",
                            f"sudo rm -f {new_binary}",
                            f"sudo mv ./{benchmark_bin} {new_binary}"
                            ]

            generate.extend(variable_gen)
            generate.append(' '.join(str(r) for v in gen for r in v))
            execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, new_bench_bin, debug=debug)


def main():
    debug_mode = False
    try:
        parameter = str(sys.argv[-1]).upper()

        if parameter == 'DEBUG':
            debug_mode = True
    except IndexError:
        debug_mode = False
    board, _ = discover_board()
    board, _ = discover_board()
    for compiler in COMPILER_VERSION:
        for flag in COMPILER_FLAGS:
            config(device=board, compiler=compiler, debug=debug_mode, flag=flag)


if __name__ == "__main__":
    main()
