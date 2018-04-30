#!/usr/bin/python2.7
import argparse
import pickle
import _log_helper as lh
import caffe
import lmdb
import numpy as np
from time import time

LOG_INTERVAL = 10
MAX_ERROR_COUNT = 1000
LENET_PRECISION = 'single'


def set_device(device):
    caffe.set_device(device)
    caffe.set_mode_gpu()


def training(solver_file):
    """
    Training function
    :param solver_file: prototxt solver
    :return: void
    """
    solver = caffe.get_solver(solver_file)
    # solver.solve() # solve completely

    number_iteration = 10000

    # collect the information
    display = 100

    # test information
    test_iteration = 100
    test_interval = 100

    # loss and accuracy information
    train_loss = np.zeros(int(np.ceil(number_iteration * 1.0 / display)))
    test_loss = np.zeros(int(np.ceil(number_iteration * 1.0 / test_interval)))
    test_accuracy = np.zeros(int(np.ceil(number_iteration * 1.0 / test_interval)))

    # tmp variables
    _test_loss = 0
    _test_accuracy = 0

    # main loop
    for iteration in range(number_iteration):
        solver.step(1)

        if 0 == iteration % display:
            train_loss[iteration // display] = solver.net.blobs['loss'].data

        if 0 == iteration % test_interval:
            for test_iter in range(test_iteration):
                solver.test_nets[0].forward()
                _test_loss += solver.test_nets[0].blobs['loss'].data
                _test_accuracy += solver.test_nets[0].blobs['accuracy'].data

            test_loss[iteration / test_interval] = _test_loss / test_iteration
            test_accuracy[iteration / test_interval] = _test_accuracy / test_iteration
            _test_loss = 0
            _test_accuracy = 0


def testing(model, weights, db_path, max_predictions=5):
    """
    Normal testing, without logs
    :param max_predictions: max number of test predictions, default = 5
    :param model: prototxt file
    :param weights: .caffemodel
    :param db_path: path to lmdb files
    :return: void
    """
    net = caffe.Net(model, weights, caffe.TEST)
    lmdb_env = lmdb.open(db_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    count = 0
    correct = 0
    for key, value in lmdb_cursor:
        count += 1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        net.blobs['data'].data[...] = np.asarray([image])
        # out = net.forward_all(data=np.asarray([image]))
        out = net.forward()

        predicted_label = out['prob'][0].argmax(axis=0)
        if label == predicted_label:
            correct += 1
        if count > max_predictions: break

    print(
        "{} out of {} were classified correctly, precision of {}".format(correct, count, float(correct) / float(count)))


def generating_radiation(model, weights, db_path, gold_path):
    """
    Generates a gold that can be used on radiation tests
    :param gold_path: gold file path
    :param model: prototxt file
    :param weights: .caffemodel file (weights)
    :param db_path: lmdb file that contais mnist test
    :return: void, it picle the dictionary using save_file
    """
    print("Generating gold for lenet " + LENET_PRECISION)
    output_list = []
    net = caffe.Net(model, weights, caffe.TEST)
    lmdb_env = lmdb.open(db_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        net.blobs['data'].data[...] = np.asarray([image])
        # out = net.forward_all(data=np.asarray([image]))
        out = net.forward()

        predicted_label = out['prob'][0].argmax(axis=0)
        correct = label == predicted_label
        output_list.append([label, predicted_label, correct])

    save_file(gold_path, output_list)
    print("Gold generate with sucess for lenet " + LENET_PRECISION)


def testing_radiation(model, weights, db_path, gold_path, iterations):
    """
    Radiation test functions
    :param model: prototxt file
    :param weights: .caffemodel file (weights)
    :param db_path: lmdb file that contains mnist test
    :param gold_path: gold filename path
    :param iterations: radiation iterations
    :return: void
    """
    string_info = "iterations: {} gold: {} dataset: mnist weights: {} model: {} db_path: {}".format(iterations,
                                                                                                    gold_path, weights,
                                                                                                    model, db_path)
    # STARTING log file
    lh.start_log_file("Lenet" + LENET_PRECISION.title(), string_info)
    lh.set_iter_interval_print(LOG_INTERVAL)

    gold_data = load_file(gold_path)
    net = caffe.Net(model, weights, caffe.TEST)
    lmdb_env = lmdb.open(db_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    overall_errors = 0
    for iteration in range(iterations):
        i = 0
        local_errors = 0
        average_time = 0.0
        for key, value in lmdb_cursor:

            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            label = int(datum.label)
            image = caffe.io.datum_to_array(datum)
            image = image.astype(np.uint8)
            net.blobs['data'].data[...] = np.asarray([image])
            lh.start_iteration()
            tic = time()
            out = net.forward()
            toc = time()
            average_time += toc - tic
            lh.end_iteration()

            if i % LOG_INTERVAL == 0:
                print("Iteration = {}, averaget time = {}, iteration errors = {}, overall errors {}"
                      .format(i, average_time / float(LOG_INTERVAL), local_errors, overall_errors))
                average_time = 0.0

            predicted_label = out['prob'][0].argmax(axis=0)
            correct = label == predicted_label
            # [label, predicted_label, correct]
            gold_label = gold_data[i][0]
            gold_predicted_label = gold_data[i][1]
            gold_correct = gold_data[i][2]

            if label != gold_label or gold_predicted_label != predicted_label or gold_correct != correct:
                error_detail = 'sample: {} label_e: {} label_r: {} predicted_label_e: {} predicted_label_r: {} '
                error_detail += 'gold_correct_e: {} gold_correct_r: {}'
                lh.log_error_detail(error_detail.format(i, gold_label, label,
                                                        gold_predicted_label, predicted_label,
                                                        gold_correct, correct))
                lh.log_error_count(1)
                overall_errors += 1
                local_errors += 1
            if overall_errors > MAX_ERROR_COUNT:
                raise ValueError("MAX ERROR COUNT REACHED")

            i += 1

    # CLOSING log file
    lh.end_log_file()


# write gold for pot use
def save_file(filename, data):
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    except Exception as err:
        print("ERROR ON SAVING FILE" + str(err))


# open gold file
def load_file(filename):
    try:
        with open(filename, "rb") as f:
            ret = pickle.load(f)
    except Exception as err:
        print("ERROR ON LOAD FILE" + str(err))
        return None
    return ret


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    # radiation logs
    parser.add_argument('--ite', dest='iterations', help="number of iterations", default=1, type=int)

    parser.add_argument('--testmode', dest='test_mode', help="0 Generates GOLD file\n"
                                                             "1 Radiation test\n"
                                                             "2 Test on mnist\n"
                                                             "3 Train on mnist\n"
                                                             "for option 2, solver file must be passed",
                        default=0, choices=[0, 1, 2, 3], type=int)

    parser.add_argument('--prototxt', dest='prototxt', help="prototxt file path",
                        default="caffe/examples/mnist/lenet_train_test.prototxt")

    parser.add_argument('--lenet_model', dest='model', help='lenet.caffemodel',
                        default='caffe/examples/mnist/lenet_iter_10000.caffemodel')

    parser.add_argument('--lmdb', dest='lmdb', help='lmdb file path, it can be test or train',
                        default='caffe/examples/mnist/mnist_test_lmdb/')

    parser.add_argument('--solver', dest='solver', help='lenet solver prototxt',
                        default='caffe/examples/mnist/lenet_solver.prototxt')

    parser.add_argument('--gold', dest='gold', help='gold file', default='./lenet_gold.gold')

    args = parser.parse_args()

    return args


def main():
    """
    MAIN FUNCTION
    :return: void
    """
    args = parse_args()
    set_device(int(args.gpu_id))
    # GENERATE CASE
    if args.test_mode == 0:
        generating_radiation(model=args.prototxt, weights=args.model, db_path=args.lmdb, gold_path=args.gold)

    elif args.test_mode == 1:  # RADIATION CASE
        testing_radiation(model=args.prototxt, weights=args.model, db_path=args.lmdb, gold_path=args.gold,
                          iterations=args.iterations)

    elif args.test_mode == 2:  # NORMAL TESTING CASE
        testing(model=args.prototxt, weights=args.model, db_path=args.lmdb)

    elif args.test_mode == 3:  # TRAIN ON MNIST
        training(solver_file=args.solver)


if __name__ == '__main__':
    main()
