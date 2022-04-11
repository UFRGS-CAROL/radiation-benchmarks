#!/usr/bin/python3

import configparser
import copy
import json
import os
import re
import socket

ALPHA = 1.0
BETA = 0.0
SIZES = [2048]
PRECISIONS = ["float"]  # , "half"]
ITERATIONS = 10000000
USE_TENSOR_CORES = [False]
USE_CUBLAS = [True]
MEMTMR = False

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


def config(device, compiler, flag):
    benchmark_bin = "gemm"
    cuda_version = compiler[0]
    cxx_version = compiler[1]
    flags_parsed = re.sub("-*=*[ ]*\"*", "", flag)

    new_bench_bin = f"{benchmark_bin}_{cuda_version}_{flags_parsed}"
    print(f"Generating {benchmark_bin} for CUDA, board:{device}")

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = configparser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise IOError("Configuration setup error: " + str(e))

    data_path = install_dir + "data/gemm"
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/gemm"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0o777)
        os.chmod(data_path, 0o777)

    generate = [f"mkdir -p {bin_path}", f"cd {src_benchmark}", "mkdir -p " + data_path]
    execute = []

    # gen only for max size, defined on cuda_trip_mxm.cu
    for precision in PRECISIONS:
        for size in SIZES:
            for use_tensor_cores in USE_TENSOR_CORES:
                for cublas in USE_CUBLAS:
                    new_binary = f"{bin_path}/{new_bench_bin}"
                    cuda_path = f"/usr/local/cuda-{cuda_version}"
                    default_path = f'size_{size}_tensor_{use_tensor_cores}_cublas_{cublas}_'
                    default_path += f'precision_{precision}_{cuda_version}_{flags_parsed}.matrix '
                    gen = [
                        [f'env LD_LIBRARY_PATH={cuda_path}/' + 'lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'],
                        [f"{new_binary}"],
                        [f'--size {size}'],
                        [f'--alpha {ALPHA} --beta {BETA}'],
                        [f'--input_a {data_path}/A_{default_path}'],
                        [f'--input_b {data_path}/B_{default_path}'],
                        [f'--input_c {data_path}/C_{default_path}'],
                        [f'--gold {data_path}/GOLD_{default_path}'],
                        [f'--tensor_cores' if use_tensor_cores else ''],
                        [f'--precision {precision}'],
                        ['--use_cublas' if cublas else ''],
                        [f'--iterations {ITERATIONS}'],
                        ['--triplicated' if MEMTMR else ''],
                    ]

                    # change mode and iterations for exe
                    exe = copy.deepcopy(gen)
                    gen.append(['--generate'])
                    # gen.append(['--check_input_existence'])
                    gen.append(['--verbose'])
                    variable_gen = ["make clean",
                                    f"make -j 4 LOGS=1 NVCCOPTFLAGS={flag} CXX={cxx_version} CUDAPATH={cuda_path}",
                                    f"rm -f {new_binary}", f"mv ./{benchmark_bin} {new_binary}"]

                    generate.extend(variable_gen)
                    generate.append(' '.join(str(r) for v in gen for r in v))
                    execute.append({
                        "killcmd": f"killall -9 {new_bench_bin}",
                        "exec": ' '.join(str(r) for v in exe for r in v),
                        "codename": new_bench_bin,
                        "header": ' '.join(str(r) for v in exe[1:] for r in v)
                    })

        with open("./cudaGEMMNewSetupConfig.json", "w") as fp:
            json.dump(execute, indent=4, fp=fp)

        for g in generate:
            if os.system(g) != 0:
                raise SystemExit(f"Could not generate gold using command:\n{g}")


def main():
    for compiler in COMPILER_VERSION:
        for flag in COMPILER_FLAGS:
            config(device=socket.gethostname(), compiler=compiler, flag=flag)


if __name__ == "__main__":
    main()
