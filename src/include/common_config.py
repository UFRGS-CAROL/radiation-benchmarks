import os
from socket import  gethostname

DEBUG_MODE = True
POSSIBLE_BOARDS_BRANDS = {"NVIDIA" : "nvidia-smi --query-gpu=gpu_name --format=csv,noheader",
                          "AMD" : "clinfo",
                          "INTEL" : ""}

def execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin):
    """
    This function will execute generate commands and create json file for the chosen benchmark
    :param execute:
    :param generate:
    :param install_dir:
    :param benchmark_bin:
    :return:
    """
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


def discover_board():
    """
    :return: the board model and the hostname
    """
    hostname = gethostname()
    for test_board in POSSIBLE_BOARDS_BRANDS:
        if os.system(test_board) != 0:
            return test_board, hostname
