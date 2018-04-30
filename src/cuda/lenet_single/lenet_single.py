caffe_root = '/home/carol/radiation-benchmarks/src/cuda/lenet_single/caffe/'  # this file should be run from {
# caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2


def test(prototxt, lenet_model, lmdb_file):
    layer_output = open("./caffe_layer_output.txt", "w")

    # load the model
    net = caffe.Net(prototxt,
                    lenet_model,
                    caffe.TEST)

    # load input and configure preprocessing
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    instance = 0
    for key, value in lmdb_cursor:
        instance += 1

        if instance <= 1:

            datum.ParseFromString(value)

            label = datum.label
            data = caffe.io.datum_to_array(datum)
            image = data.astype(np.float32)

            dim1 = len(image)
            dim2 = len(image[0])
            dim3 = len(image[0][0])
            for i in range(0, dim1):
                for j in range(0, dim2):
                    for k in range(0, dim3):
                        image[i][j][k] *= 0.00390625

            # load the image in the data layer
            net.blobs['data'].data[...] = image

            # compute
            # out = net.forward()
            out = net.forward(start='data', end='ip2')

            # predicted predicted class
            # print out['prob'].argmax()
            print out['ip2']
            layer_results = out['ip2']

            dim1 = len(layer_results)
            dim2 = len(layer_results[0])
            # dim3 = len(layer_results[0][0])
            # dim4 = len(layer_results[0][0][0])

            # print(str(dim1) + " | " + str(dim2) + " | " + str(dim3) + " | " + str(dim4) + "\n\n\n")

            for i in range(0, dim1):
                for j in range(0, dim2):
                    layer_output.write(str(layer_results[i][j]) + "\n")
                    # for k in range(0, dim3):
                    # for l in range(0, dim4):
                    # layerOutput.write(str(layer_results[i][j][k][l]) + "\n")


def set_device():
    caffe.set_device(0)
    caffe.set_mode_gpu()


def training(solver_file):
    """
    training
    """

    set_device()

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
    _train_loss = 0;
    _test_loss = 0;
    _test_accuracy = 0;

    # main loop
    for iter in range(number_iteration):
        solver.step(1)

        # save model during training
        # ~ if iter == number_iteration - 1: #in [10, 30, 60, 100, 300, 600, 1000, 3000, 6000, number_iteration - 1]:
        # ~ string = 'lenet_iter_%(iter)d.caffemodel'%{'iter': iter}
        # ~ solver.net.save(string)

        if 0 == iter % display:
            train_loss[iter // display] = solver.net.blobs['loss'].data

        '''
        # accumulate the train loss
        _train_loss += solver.net.blobs['SoftmaxWithLoss1'].data

        if 0 == iter % display:
            train_loss[iter // display] = _train_loss / display
            _train_loss = 0
        '''

        if 0 == iter % test_interval:
            for test_iter in range(test_iteration):
                solver.test_nets[0].forward()
                _test_loss += solver.test_nets[0].blobs['loss'].data
                _test_accuracy += solver.test_nets[0].blobs['accuracy'].data

            test_loss[iter / test_interval] = _test_loss / test_iteration
            test_accuracy[iter / test_interval] = _test_accuracy / test_iteration
            _test_loss = 0
            _test_accuracy = 0


def test_model(model, weights, db_path):
    net = caffe.Net(model, weights, caffe.TEST)
    set_device()
    # db_path = './examples/mnist/mnist_test_lmdb'

    lmdb_env = lmdb.open(db_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    count = 0
    correct = 0
    for key, value in lmdb_cursor:
        print "Count:", count
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
        print(predicted_label)
        if label == predicted_label[0][0]:
            correct += 1
        print("Label is class " + str(label) + ", predicted class is " + str(predicted_label[0][0]))
        if count == 3:
            break
    print(str(correct) + " out of " + str(count) + " were classified correctly")


# Before keep going execute
# data/mnist/get_mnist.sh
# examples/mnist/create_mnist.sh

if __name__ == '__main__':
    solver = "/home/carol/radiation-benchmarks/src/cuda/lenet_single/caffe/examples/mnist/lenet_solver.prototxt"
    model = "/home/carol/radiation-benchmarks/src/cuda/lenet_single/caffe/examples/mnist/lenet_train_test.prototxt"
    weights = "/home/carol/radiation-benchmarks/src/cuda/lenet_single/caffe/examples/mnist/lenet_iter_10000.caffemodel"
    db_train_path = "/home/carol/radiation-benchmarks/src/cuda/lenet_single/caffe/examples/mnist/mnist_train_lmdb/"
    db_test_path = "/home/carol/radiation-benchmarks/src/cuda/lenet_single/caffe/examples/mnist/mnist_test_lmdb/"

    # training(solver_file=solver)
    lenet_model_prototxt = "/home/carol/radiation-benchmarks/src/cuda/lenet_single/caffe/examples/mnist/lenet.prototxt"
    test_model(lenet_model_prototxt, weights, db_test_path)

