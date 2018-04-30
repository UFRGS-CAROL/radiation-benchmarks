#!/usr/bin/python

import os
import sys
import ConfigParser
import copy
from errno import ENOENT

DEBUG_MODE = False
LENET_PRECISION = 'single'
DATASETS = [
    # normal
    {'train_model': 'caffe/examples/mnist/lenet_train_test.prototxt',
     'test_model': 'caffe/examples/mnist/lenet.prototxt',
     'weights': 'caffe/examples/mnist/lenet_iter_10000.caffemodel',
     'gold': 'gold_lenet_' + LENET_PRECISION + '_10k.gold',
     'solver': 'caffe/examples/mnist/lenet_solver.prototxt',
     'db_train_path': 'caffe/examples/mnist/mnist_train_lmdb/',
     'db_test_path': 'caffe/examples/mnist/mnist_test_lmdb/'},

]

CAFFE_PYTHON = 'caffe/python'
LOG_HELPER_LIB = 'include/log_helper_swig_wraper'


def create_mnist(src_lenet):
    """
    Execute caffe scripts to create mnist
    caffe/data/mnist/get_mnist.sh
    caffe/examples/mnist/create_mnist.sh
    :param src_lenet: where lenet is located
    :return: void, raise an exception if MNIST weren't created
    """
    dataset = DATASETS[0]
    a_files = [dataset['db_train_path'] + "data.mdb", dataset['db_train_path'] + "lock.mdb",
               dataset['db_test_path'] + "data.mdb", dataset['db_test_path'] + "lock.mdb",
               src_lenet + '/caffe/data/mnist/get_mnist.sh', src_lenet + '/caffe/examples/mnist/create_mnist.sh']

    a_exist = [f for f in a_files if os.path.isfile(f)]
    a_non_exist = list(set(a_exist) ^ set(a_files))
    if len(a_non_exist) != 0:
        raise IOError(ENOENT, 'NOT A FILE', a_non_exist[0])

    if len(a_exist) == len(a_files):
        return

    for e in [src_lenet + '/caffe/data/mnist/get_mnist.sh', src_lenet + '/caffe/examples/mnist/create_mnist.sh']:
        if os.system(e) != 0:
            raise ValueError("Something went wrong when executing: ", str(e))


def main(board):
    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)

        install_dir = config.get('DEFAULT', 'installdir')

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    benchmark_bin = 'lenet_' + LENET_PRECISION + '.py'
    data_path = install_dir + "/data/lenet"
    src_lenet = install_dir + "/src/cuda/lenet_" + LENET_PRECISION
    bin_path = src_lenet + "/" + benchmark_bin

    print "Generating " + benchmark_bin + " precision for CUDA on " + str(board)

    if not DEBUG_MODE and not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    # check if all training files are ok
    if not DEBUG_MODE:
        create_mnist(src_lenet=src_lenet)

    # Insert caffe and log helper to PYTHONPATH
    env_command = "PYTHONPATH=" + src_lenet + "/" + CAFFE_PYTHON
    env_command += ":" + install_dir + "/src/" + LOG_HELPER_LIB + ":$PYTHONPATH"
    generate = ["cd " + src_lenet,
                "make -C ../../include/log_helper_swig_wraper/ log_helper_python"]
    execute = []

    for set in DATASETS:
        gen = [None] * 8
        gen[0] = ['sudo env ' + env_command, bin_path + " "]
        gen[1] = [' --ite', 1]
        gen[2] = [' --testmode ', 0]
        gen[3] = [' --ite ', '1']
        gen[4] = ['--prototxt ', src_lenet + "/" + set['test_model']]
        gen[5] = ['--lenet_model ', src_lenet + "/" + set['weights']]
        gen[6] = ['--lmdb ', src_lenet + "/" + set['db_test_path']]
        gen[7] = ['--gold ', data_path + '/' + set['gold']]

        exe = copy.deepcopy(gen)
        exe[2][1] = 1
        exe[3][1] = 1000

        generate.append(' '.join(str(r) for v in gen for r in v))
        execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute=execute, generate=generate, install_dir=install_dir,
                                   benchmark_bin=benchmark_bin)


def execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin):
    for i in generate:
        print i
        if not DEBUG_MODE:
            if os.system(str(i)) != 0:
                print "Something went wrong with generate of ", str(i)
                exit(1)

    list_to_print = ["[\n"]
    for ii, i in enumerate(execute):
        command = "{\"killcmd\": \"pkill -9 " + benchmark_bin + "; kiall -9 " + benchmark_bin + "\", \"exec\": \"" + str(
            i) + "\"}"
        if ii != len(execute) - 1:
            command += ',\n'
        list_to_print.append(command)
    list_to_print.append("\n]")
    new_bench_bin = benchmark_bin.replace(".py", "")
    if not DEBUG_MODE:
        with open(install_dir + "/scripts/json_files/" + new_bench_bin + ".json", 'w') as fp:
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
