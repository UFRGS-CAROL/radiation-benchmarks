caffe_root = '/home/carol/radiation-benchmarks/src/cuda/LeNetCUDA/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import lmdb
import numpy as np
from caffe import layers as L, params as P, proto, to_proto


# file path
root = '/home/your-account/DL-Analysis/'
train_list = root + 'mnist/mnist_train_lmdb'
test_list = root + 'mnist/mnist_test_lmdb'

train_proto = root + 'mnist/LeNet/train.prototxt'
test_proto = root + 'mnist/LeNet/test.prototxt'

deploy_proto = root + 'mnist/LeNet/deploy.prototxt'

solver_proto = root + 'mnist/LeNet/solver.prototxt'

def LeNet(data_list, batch_size, IncludeAccuracy = False, deploy = False):
    """
    LeNet define
    """

    if not(deploy):
        data, label =  L.Data(source = data_list,
                              backend = P.Data.LMDB,
                              batch_size = batch_size,
                              ntop = 2,
                              transform_param = dict(scale = 0.00390625))
    else:
        data = L.Input(input_param = {'shape': {'dim': [64, 1, 28, 28]}})
    
    conv1 = L.Convolution(data,
                          kernel_size = 5,
                          stride = 1,
                          num_output = 20,
                          pad = 0,
                          weight_filler = dict(type = 'xavier'))
    
    pool1 = L.Pooling(conv1,
                      pool = P.Pooling.MAX,
                      kernel_size = 2,
                      stride = 2)
    
    conv2 = L.Convolution(pool1,
                          kernel_size = 5,
                          stride = 1,
                          num_output = 50,
                          pad = 0,
                          weight_filler = dict(type = 'xavier'))
    
    pool2 = L.Pooling(conv2,
                      pool = P.Pooling.MAX,
                      kernel_size = 2,
                      stride = 2)

    ip1 = L.InnerProduct(pool2,
                         num_output = 500,
                         weight_filler = dict(type = 'xavier'))
    
    relu1 = L.ReLU(ip1,
                   in_place = True)
    
    ip2 = L.InnerProduct(relu1,
                         num_output = 10,
                         weight_filler = dict(type = 'xavier'))
    
    #loss = L.SoftmaxWithLoss(ip2, label)

    if ( not(IncludeAccuracy) and not(deploy) ):
        # train net
        loss = L.SoftmaxWithLoss(ip2, label)
        return to_proto(loss)
    
    elif ( IncludeAccuracy and not(deploy) ):
        # test net
        loss = L.SoftmaxWithLoss(ip2, label)
        Accuracy = L.Accuracy(ip2, label)
        return to_proto(loss, Accuracy)
    
    else:
        # deploy net
        prob = L.Softmax(ip2)
        return to_proto(prob)
    
def WriteNet():
    """
    write proto to file
    """
    
    # train net
    with open(train_proto, 'w') as file:
        file.write( str(LeNet(train_list, 64, IncludeAccuracy = False, deploy = False)) )

    # test net
    with open(test_proto, 'w') as file:
        file.write( str(LeNet(test_list, 100, IncludeAccuracy = True, deploy = False)) )

    # deploy net
    with open(deploy_proto, 'w') as file:
        file.write( str(LeNet('not need', 64, IncludeAccuracy = False, deploy = True)) )

def GenerateSolver(solver_file, train_net, test_net):
    """
    generate the solver file
    """
    
    s = proto.caffe_pb2.SolverParameter()
    s.train_net = train_net
    s.test_net.append(test_net)
    s.test_interval = 100
    s.test_iter.append(100)
    s.max_iter = 10000
    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.lr_policy = 'step'
    s.stepsize = 3000
    s.gamma = 0.1
    s.display = 100
    s.snapshot = 0
    s.snapshot_prefix = './lenet'
    s.type = 'SGD'
    s.solver_mode = proto.caffe_pb2.SolverParameter.GPU

    with open(solver_file, 'w') as file:
        file.write( str(s) )

def set_device():
    caffe.set_device(0)
    caffe.set_mode_gpu()

def Training(solver_file):
    """
    training
    """
 
    set_device()
    
    solver = caffe.get_solver(solver_file)
    #solver.solve() # solve completely
    
    number_iteration = 10000

    # collect the information
    display = 100

    # test information
    test_iteration = 100
    test_interval = 100

    # loss and accuracy information
    train_loss = np.zeros( int(np.ceil(number_iteration * 1.0 / display)) )
    test_loss = np.zeros( int(np.ceil(number_iteration * 1.0 / test_interval)) )
    test_accuracy = np.zeros( int(np.ceil(number_iteration * 1.0 / test_interval)) )

    # tmp variables
    _train_loss = 0; _test_loss = 0; _test_accuracy = 0;

    # main loop
    for iter in range(number_iteration):
        solver.step(1)

        # save model during training
        #~ if iter == number_iteration - 1: #in [10, 30, 60, 100, 300, 600, 1000, 3000, 6000, number_iteration - 1]:
            #~ string = 'lenet_iter_%(iter)d.caffemodel'%{'iter': iter}
            #~ solver.net.save(string)

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

    # save for analysis
    #~ np.save('./train_loss.npy', train_loss)
    #~ np.save('./test_loss.npy', test_loss)
    #~ np.save('./test_accuracy.npy', test_accuracy)
    
def test(model, weights, db_path):
    net = caffe.Net(model, weights,caffe.TEST)
    set_device()
    #db_path = './examples/mnist/mnist_test_lmdb'
    
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
        #out = net.forward_all(data=np.asarray([image]))
        out = net.forward()
        
        predicted_label = out['prob'][0].argmax(axis=0)
        if label == predicted_label[0][0]:
            correct = correct + 1
        print("Label is class " + str(label) + ", predicted class is " + str(predicted_label[0][0]))
        if count == 3:
            break
    print(str(correct) + " out of " + str(count) + " were classified correctly")
    
    
    
# Before keep going execute
# data/mnist/get_mnist.sh
# examples/mnist/create_mnist.sh
   
if __name__ == '__main__':
    #~ WriteNet()
    #~ GenerateSolver(solver_proto, train_proto, test_proto)
    solver = "/home/carol/radiation-benchmarks/src/cuda/LeNetCUDA/caffe/examples/mnist/lenet_solver.prototxt"
    model = "/home/carol/radiation-benchmarks/src/cuda/LeNetCUDA/caffe/examples/mnist/lenet_train_test.prototxt"
    weights = "/home/carol/radiation-benchmarks/src/cuda/LeNetCUDA/caffe/examples/mnist/lenet_iter_10000.caffemodel"
    db_path = "/home/carol/radiation-benchmarks/src/cuda/LeNetCUDA/caffe/examples/mnist/mnist_test_lmdb/"
    #Training(model)
    test(model, weights, db_path)
