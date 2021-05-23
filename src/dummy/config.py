#!/usr/bin/python3

import configparser
import copy
import os
import sys

sys.path.append("../include")
from common_config import discover_board, execute_and_write_json_to_file


def config(device, debug):
    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = configparser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise IOError("Configuration setup error: " + str(e))

    data_path = install_dir + "data/dummy"
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/dummy"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0o777)
        os.chmod(data_path, 0o777)

    new_benchmark_bin =  f"{bin_path}/dummy"
    generate = [f"sudo mkdir -p {bin_path}",
                f"cd {src_benchmark}",
                "make -C ../include ",
                "make",  f"mv dummy {new_benchmark_bin}"
                ]
                
    execute = [ f'sudo {new_benchmark_bin}']
    execute_and_write_json_to_file(execute, generate, install_dir, "dummy", debug=debug)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[-1]).upper()

        if parameter == 'DEBUG':
            debug_mode = True
    except IndexError:
        debug_mode = False
 
    config(device="dummy", debug=debug_mode)
