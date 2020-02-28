import os
from socket import gethostname

#ADD new boards lines here
POSSIBLE_BOARDS_BRANDS = {"NVIDIA": "nvidia-smi --query-gpu=gpu_name --format=csv,noheader",
                          "AMD": "clinfo",
                          "INTEL": "something_here",
                          "TX1": "cat /etc/nv_tegra_release",
                          "TX2": "cat /etc/nv_tegra_release"}


def execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug):
    """
    This function will execute generate commands and create json file for the chosen benchmark
    :param execute:
    :param generate:
    :param install_dir:
    :param benchmark_bin:
    :param debug: if you want debug the config
    :return:
    """
    for i in generate:
        print(i)
        if not debug and os.system(str(i)) != 0:
                print("Something went wrong with generate of ", str(i))
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

    print("\nConfiguring done, to run check file: " + install_dir + "scripts/json_files/" + benchmark_bin + ".json")


def discover_board():
    """
    :return: the board model and the hostname
    """
    hostname = gethostname()
    for test_board, test_command in POSSIBLE_BOARDS_BRANDS.items():
        if os.system(test_command + " 2> /tmp/config") == 0:
            return test_board, hostname
    return None, None
