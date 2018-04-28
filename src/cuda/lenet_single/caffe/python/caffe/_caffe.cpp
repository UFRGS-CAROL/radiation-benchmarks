#include <Python.h>  // NOLINT(build/include_alpha)

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <google/protobuf/text_format.h>
#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/python_layer.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/gpu_memory.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

// Hack to convert macro to string
#define STRINGIZE(m) #m
#define STRINGIZE2(m) STRINGIZE(m)

/* Fix to avoid registration warnings in pycaffe (#3960) */
#define BP_REGISTER_SHARED_PTR_TO_PYTHON(PTR) do { \
  const boost::python::type_info info = \
    boost::python::type_id<shared_ptr<PTR > >(); \
  const boost::python::converter::registration* reg = \
    boost::python::converter::registry::query(info); \
  if (reg == NULL) { \
    bp::register_ptr_to_python<shared_ptr<PTR > >(); \
  } else if ((*reg).m_to_python == NULL) { \
    bp::register_ptr_to_python<shared_ptr<PTR > >(); \
  } \
} while (0)

namespace bp = boost::python;

namespace caffe {

// For Python, for now, we'll just always use float as the type.
typedef float Dtype;
const int NPY_DTYPE = NPY_FLOAT32;
shared_ptr<GPUMemory::Scope> gpu_memory_scope;

void initialize_gpu_memory_scope(const vector<int>& gpus) {
  FLAGS_alsologtostderr = 1;
  static bool google_initialized = false;
  if (!google_initialized) {
    google_initialized = true;
    google::InitGoogleLogging("pyNVCaffe");
  }
  if (!gpu_memory_scope) {
    gpu_memory_scope.reset(new GPUMemory::Scope(gpus));
    if (gpus.size() > 0) {
      Caffe::SetDevice(gpus[0]);
    }
  }
  if (gpus.size() > 0) {
    Caffe::set_gpus(gpus);
  }
}

// Selecting mode.
void set_mode_cpu() {
  Caffe::set_mode(Caffe::CPU);
  // We need to run GPU-built Caffe on CPU sometimes.
  vector<int> gpus(1, 0);
  initialize_gpu_memory_scope(gpus);
}

void set_mode_gpu() {
  PyGILRelease gil;
  Caffe::set_mode(Caffe::GPU);
  vector<int> gpus(1, 0);
  initialize_gpu_memory_scope(gpus);
}

void set_device(int gpu) {
  CHECK_GE(gpu, 0);
  Caffe::set_mode(Caffe::GPU);
  vector<int> gpus(1, gpu);
  initialize_gpu_memory_scope(gpus);
}

void set_devices(const bp::list& lst) {
  CHECK(!lst.is_none());
  boost::python::ssize_t len = boost::python::len(lst);
  CHECK(len);
  vector<int> gpus(len);
  for (int i = 0; i < len; ++i) {
    gpus[i] = boost::python::extract<int>(lst[i]);
    CHECK_GE(gpus[i], 0);
  }
  Caffe::set_mode(Caffe::GPU);
  initialize_gpu_memory_scope(gpus);
}

// For convenience, check that input files can be opened, and raise an
// exception that boost will send to Python if not (caffe could still crash
// later if the input files are disturbed before they are actually used, but
// this saves frustration in most cases).
static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    if (!f.good()) {
      f.close();
      throw std::runtime_error("Could not open file " + filename);
    }
    f.close();
}

void CheckContiguousArray(PyArrayObject* arr, string name,
    int channels, int height, int width) {
  if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
    throw std::runtime_error(name + " must be C contiguous");
  }
  if (PyArray_NDIM(arr) != 4) {
    throw std::runtime_error(name + " must be 4-d");
  }
  if (PyArray_TYPE(arr) != NPY_FLOAT32) {
    throw std::runtime_error(name + " must be float32");
  }
  if (PyArray_DIMS(arr)[1] != channels) {
    throw std::runtime_error(name + " has wrong number of channels");
  }
  if (PyArray_DIMS(arr)[2] != height) {
    throw std::runtime_error(name + " has wrong height");
  }
  if (PyArray_DIMS(arr)[3] != width) {
    throw std::runtime_error(name + " has wrong width");
  }
}

// Net constructor for passing phase as int
shared_ptr<Net> Net_Init(string param_file, int phase,
    const int level, const bp::object& stages,
    const bp::object& weights) {
  // Convert stages from list to vector
  vector<string> stages_vector;
  if (!stages.is_none()) {
    for (int i = 0; i < bp::len(stages); i++) {
      stages_vector.push_back(bp::extract<string>(stages[i]));
    }
  }
  PyGILRelease gil;
  CheckFile(param_file);
  shared_ptr<Net> net(new Net(param_file, static_cast<Phase>(phase),
      0U, nullptr, nullptr, false, level, &stages_vector));
  // Load weights
  if (!weights.is_none()) {
    std::string weights_file_str = bp::extract<std::string>(weights);
    CheckFile(weights_file_str);
    net->CopyTrainedLayersFrom(weights_file_str);
  }
  return net;
}

// Net construct-and-load convenience constructor
shared_ptr<Net> Net_Init_Load(string param_file, string pretrained_param_file, int phase) {
  PyGILRelease gil;
  LOG(WARNING) << "DEPRECATION WARNING - deprecated use of Python interface";
  LOG(WARNING) << "Use this instead (with the named \"weights\""
    << " parameter):";
  LOG(WARNING) << "Net('" << param_file << "', " << phase
    << ", weights='" << pretrained_param_file << "')";
  CheckFile(param_file);
  CheckFile(pretrained_param_file);
  shared_ptr<Net> net(new Net(param_file, static_cast<Phase>(phase)));
  net->CopyTrainedLayersFrom(pretrained_param_file);
  return net;
}

void Net_Save(const Net& net, string filename) {
  NetParameter net_param;
  net.ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, filename.c_str());
}

void Net_SetInputArrays(Net* net, bp::object data_obj, bp::object labels_obj) {
  // check that this network has an input MemoryDataLayer
  shared_ptr<MemoryDataLayer<Dtype, Dtype> > md_layer =
    boost::dynamic_pointer_cast<MemoryDataLayer<Dtype, Dtype> >(net->layers()[0]);
  if (!md_layer) {
    throw std::runtime_error("set_input_arrays may only be called if the"
        " first layer is a MemoryDataLayer");
  }

  // check that we were passed appropriately-sized contiguous memory
  PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
  PyArrayObject* labels_arr = reinterpret_cast<PyArrayObject*>(labels_obj.ptr());
  CheckContiguousArray(data_arr, "data array", md_layer->channels(),
      md_layer->height(), md_layer->width());
  CheckContiguousArray(labels_arr, "labels array", 1, 1, 1);
  if (PyArray_DIMS(data_arr)[0] != PyArray_DIMS(labels_arr)[0]) {
    throw std::runtime_error("data and labels must have the same first dimension");
  }
  if (PyArray_DIMS(data_arr)[0] % md_layer->batch_size() != 0) {
    throw std::runtime_error("first dimensions of input arrays must be a"
        " multiple of batch size");
  }

  md_layer->Reset(static_cast<Dtype*>(PyArray_DATA(data_arr)),
      static_cast<Dtype*>(PyArray_DATA(labels_arr)),
      PyArray_DIMS(data_arr)[0]);
}

float Net_ForwardFromToNoGIL(Net* net, int start, int end) {
  PyGILRelease gil;
  return net->ForwardFromTo(start, end);
}

void Net_BackwardFromToNoGIL(Net* net, int start, int end) {
  PyGILRelease gil;
  net->BackwardFromTo(start, end);
}

Solver* GetSolverFromFile(const string& filename) {
  SolverParameter param = ReadSolverParamsFromTextFileOrDie(filename);
  return SolverRegistry::CreateSolver(param);
}

struct NdarrayConverterGenerator {
  template <typename T> struct apply;
};

template <>
struct NdarrayConverterGenerator::apply<Dtype*> {
  struct type {
    PyObject* operator() (Dtype* data) const {
      // Just store the data pointer, and add the shape information in postcall.
      return PyArray_SimpleNewFromData(0, NULL, NPY_DTYPE, data);
    }
    const PyTypeObject* get_pytype() {
      return &PyArray_Type;
    }
  };
};

struct NdarrayCallPolicies : public bp::default_call_policies {
  typedef NdarrayConverterGenerator result_converter;
  PyObject* postcall(PyObject* pyargs, PyObject* result) {
    bp::object pyblob = bp::extract<bp::tuple>(pyargs)()[0];
    shared_ptr<Blob> blob =
      bp::extract<shared_ptr<Blob> >(pyblob);
    // Free the temporary pointer-holding array, and construct a new one with
    // the shape information from the blob.
    void* data = PyArray_DATA(reinterpret_cast<PyArrayObject*>(result));
    Py_DECREF(result);
    const int num_axes = blob->num_axes();
    vector<npy_intp> dims(blob->shape().begin(), blob->shape().end());
    PyObject *arr_obj = PyArray_SimpleNewFromData(num_axes, dims.data(),
                                                  NPY_FLOAT32, data);
    // SetBaseObject steals a ref, so we need to INCREF.
    Py_INCREF(pyblob.ptr());
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(arr_obj),
        pyblob.ptr());
    return arr_obj;
  }
};

bp::object Blob_Reshape(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("Blob.reshape takes no kwargs");
  }
  Blob* self = bp::extract<Blob*>(args[0]);
  vector<int> shape(bp::len(args) - 1);
  for (int i = 1; i < bp::len(args); ++i) {
    shape[i - 1] = bp::extract<int>(args[i]);
  }
  self->Reshape(shape);
  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}

bp::object BlobVec_add_blob(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("BlobVec.add_blob takes no kwargs");
  }
  typedef vector<shared_ptr<Blob> > BlobVec;
  BlobVec* self = bp::extract<BlobVec*>(args[0]);
  vector<int> shape(bp::len(args) - 1);
  for (int i = 1; i < bp::len(args); ++i) {
    shape[i - 1] = bp::extract<int>(args[i]);
  }
  self->push_back(shared_ptr<Blob>(new TBlob<Dtype>(shape)));
  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}


using Layer2 = Layer<Dtype, Dtype>;

// Layer
template <class T>
vector<T> py_to_vector(bp::object pyiter) {
  vector<T> vec;
  for (int i = 0; i < bp::len(pyiter); ++i) {
    vec.push_back(bp::extract<T>(pyiter[i]));
  }
  return vec;
}

// TODO unify those 8 functions:
void LayerBase_SetUp(LayerBase *layer, bp::object py_bottom, bp::object py_top) {
  vector<Blob*> bottom = py_to_vector<Blob*>(py_bottom);
  vector<Blob*> top = py_to_vector<Blob*>(py_top);
  PyGILRelease gil;
  layer->SetUp(bottom, top);
}

void LayerBase_Reshape(LayerBase *layer, bp::object py_bottom, bp::object py_top) {
  vector<Blob*> bottom = py_to_vector<Blob*>(py_bottom);
  vector<Blob*> top = py_to_vector<Blob*>(py_top);
  PyGILRelease gil;
  layer->Reshape(bottom, top);
}

Dtype LayerBase_Forward(LayerBase *layer, bp::object py_bottom, bp::object py_top) {
  vector<Blob*> bottom = py_to_vector<Blob*>(py_bottom);
  vector<Blob*> top = py_to_vector<Blob*>(py_top);
  PyGILRelease gil;
  return layer->Forward(bottom, top);
}

void LayerBase_Backward(LayerBase *layer, bp::object py_top, bp::object py_propagate_down,
    bp::object py_bottom) {
  vector<Blob*> top = py_to_vector<Blob*>(py_top);
  vector<bool> propagate_down = py_to_vector<bool>(py_propagate_down);
  vector<Blob*> bottom = py_to_vector<Blob*>(py_bottom);
  PyGILRelease gil;
  layer->Backward(top, propagate_down, bottom);
}

void Layer_SetUp(Layer2 *layer, bp::object py_bottom, bp::object py_top) {
  vector<Blob*> bottom = py_to_vector<Blob*>(py_bottom);
  vector<Blob*> top = py_to_vector<Blob*>(py_top);
  PyGILRelease gil;
  layer->SetUp(bottom, top);
}

void Layer_Reshape(Layer2 *layer, bp::object py_bottom, bp::object py_top) {
  vector<Blob*> bottom = py_to_vector<Blob*>(py_bottom);
  vector<Blob*> top = py_to_vector<Blob*>(py_top);
  PyGILRelease gil;
  layer->Reshape(bottom, top);
}

Dtype Layer_Forward(Layer2 *layer, bp::object py_bottom, bp::object py_top) {
  vector<Blob*> bottom = py_to_vector<Blob*>(py_bottom);
  vector<Blob*> top = py_to_vector<Blob*>(py_top);
  PyGILRelease gil;
  return layer->Forward(bottom, top);
}

void Layer_Backward(Layer2 *layer, bp::object py_top, bp::object py_propagate_down,
    bp::object py_bottom) {
  vector<Blob*> top = py_to_vector<Blob*>(py_top);
  vector<bool> propagate_down = py_to_vector<bool>(py_propagate_down);
  vector<Blob*> bottom = py_to_vector<Blob*>(py_bottom);
  PyGILRelease gil;
  layer->Backward(top, propagate_down, bottom);
}

// LayerParameter
shared_ptr<LayerParameter> LayerParameter_Init(bp::object py_layer_param) {
  shared_ptr<LayerParameter> layer_param(new LayerParameter);
  if (PyObject_HasAttrString(py_layer_param.ptr(), "SerializeToString")) {
    string dump = bp::extract<string>(py_layer_param.attr("SerializeToString")());
    layer_param->ParseFromString(dump);
  } else {
    try {
      string dump = bp::extract<string>(py_layer_param);
      google::protobuf::TextFormat::ParseFromString(dump, layer_param.get());
    } catch(...) {
      throw std::runtime_error("1st arg must be LayerPrameter or string.");
    }
  }
  if (!layer_param->IsInitialized()) {
    throw std::runtime_error("LayerParameter not initialized: Missing required fields.");
  }
  return layer_param;
}

void LayerParameter_FromPython(LayerParameter *layer_param, bp::object py_layer_param) {
  shared_ptr<LayerParameter> copy = LayerParameter_Init(py_layer_param);
  layer_param->Clear();
  layer_param->CopyFrom(*copy);
}

bp::object LayerParameter_ToPython(const LayerParameter *layer_param, bp::object py_layer_param) {
  string dump;
  layer_param->SerializeToString(&dump);
  py_layer_param.attr("ParseFromString")(bp::object(dump));
  return py_layer_param;
}

// Create layer from caffe_pb2.LayerParameter in Python
shared_ptr<LayerBase> create_layer(bp::object py_layer_param) {
  shared_ptr<LayerParameter> layer_param(LayerParameter_Init(py_layer_param));
  return LayerRegistry::CreateLayer(*layer_param, 0UL);
}

// Run solver step without GIL
void Solver_StepNoGIL(Solver* solver, int iters) {
  PyGILRelease gil;
  solver->Step(iters);
}


BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolveOverloads, Solve, 0, 1);

BOOST_PYTHON_MODULE(_caffe) {
  // below, we prepend an underscore to methods that will be replaced
  // in Python

  bp::scope().attr("CAFFE_VERSION") = STRINGIZE2(CAFFE_VERSION);

  // Caffe utility functions
  bp::def("set_mode_cpu", &set_mode_cpu);
  bp::def("set_mode_gpu", &set_mode_gpu);
  bp::def("set_devices", &set_devices);
  bp::def("set_device", &set_device);

  bp::def("layer_type_list", &LayerRegistry::LayerTypeList);

  bp::class_<Net, shared_ptr<Net>, boost::noncopyable >("Net",
    bp::no_init)
    // Constructor
    .def("__init__", bp::make_constructor(&Net_Init,
          bp::default_call_policies(), (bp::arg("network_file"), "phase",
            bp::arg("level")=0, bp::arg("stages")=bp::object(),
            bp::arg("weights")=bp::object())))
    // Legacy constructor
    .def("__init__", bp::make_constructor(&Net_Init_Load))
    .def("_forward", &Net_ForwardFromToNoGIL)
    .def("_backward", &Net_BackwardFromToNoGIL)
    .def("reshape", &Net::Reshape)
    .def("clear_param_diffs", &Net::ClearParamDiffs)
    // The cast is to select a particular overload.
    .def("copy_from", static_cast<void (Net::*)(const string)>(
        &Net::CopyTrainedLayersFrom))
    .def("share_with", &Net::ShareTrainedLayersWith)
    .add_property("_blob_loss_weights", bp::make_function(
        &Net::blob_loss_weights, bp::return_internal_reference<>()))
    .def("_bottom_ids", bp::make_function(&Net::bottom_ids,
        bp::return_value_policy<bp::copy_const_reference>()))
    .def("_top_ids", bp::make_function(&Net::top_ids,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_blobs", bp::make_function(&Net::blobs,
        bp::return_internal_reference<>()))
    .add_property("layers", bp::make_function(&Net::layers,
        bp::return_internal_reference<>()))
    .add_property("_blob_names", bp::make_function(&Net::blob_names,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_layer_names", bp::make_function(&Net::layer_names,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_inputs", bp::make_function(&Net::input_blob_indices,
        bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("_outputs",
        bp::make_function(&Net::output_blob_indices,
        bp::return_value_policy<bp::copy_const_reference>()))
    .def("_set_input_arrays", &Net_SetInputArrays,
        bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >())
    .def("save", &Net_Save);
  BP_REGISTER_SHARED_PTR_TO_PYTHON(Net);


  bp::class_<Blob, shared_ptr<TBlob<Dtype>>, boost::noncopyable>(
    "Blob", bp::no_init)
        .add_property("shape",
            bp::make_function(
                static_cast<const vector<int>& (Blob::*)() const>(
                    &Blob::shape),
                bp::return_value_policy<bp::copy_const_reference>()))
        .add_property("num",      &Blob::num)
        .add_property("channels", &Blob::channels)
        .add_property("height",   &Blob::height)
        .add_property("width",    &Blob::width)
        .add_property("count",    static_cast<int (Blob::*)() const>(
            &Blob::count))
        .def("reshape",           bp::raw_function(&Blob_Reshape))
        .add_property("data",     bp::make_function(&Blob::mutable_cpu_data<Dtype>,
              NdarrayCallPolicies()))
        .add_property("diff",     bp::make_function(&Blob::mutable_cpu_diff<Dtype>,
              NdarrayCallPolicies()));

  BP_REGISTER_SHARED_PTR_TO_PYTHON(Blob);

  bp::class_<TBlob<Dtype>, bp::bases<Blob>,
    shared_ptr<TBlob<Dtype>>, boost::noncopyable>("TBlob", bp::no_init);
  BP_REGISTER_SHARED_PTR_TO_PYTHON(TBlob<Dtype>);

  bp::class_<LayerBase, shared_ptr<LayerBase>, boost::noncopyable>(
    "LayerBase", bp::no_init)
    .add_property("type", bp::make_function(&LayerBase::type))
    .def("SetUp", &LayerBase_SetUp)
    .def("Reshape", &LayerBase_Reshape)
    .def("Forward", &LayerBase_Forward)
    .def("Backward", &LayerBase_Backward)
    .add_property("blobs", bp::make_function(&LayerBase::blobs,
          bp::return_internal_reference<>()));
  BP_REGISTER_SHARED_PTR_TO_PYTHON(LayerBase);

  bp::class_<Layer<Dtype, Dtype>, bp::bases<LayerBase>, shared_ptr<PythonLayer<Dtype, Dtype> >,
    boost::noncopyable>(
    "Layer", bp::init<const LayerParameter&>())
    .def("setup", &Layer<Dtype, Dtype>::LayerSetUp)
    .def("SetUp", &Layer_SetUp)
    .def("reshape", &Layer<Dtype, Dtype>::Reshape)
    .def("Reshape", &Layer_Reshape)
    .def("Forward", &Layer_Forward)
    .def("Backward", &Layer_Backward)
    .add_property("type", bp::make_function(&Layer<Dtype, Dtype>::type));
  BP_REGISTER_SHARED_PTR_TO_PYTHON(Layer2);

  bp::class_<LayerParameter, shared_ptr<LayerParameter> >("LayerParameter", bp::no_init)
    .def("__init__", bp::make_constructor(&LayerParameter_Init))
    .def("from_python", &LayerParameter_FromPython)
    .def("_to_python", &LayerParameter_ToPython);

  bp::def("create_layer", &create_layer);

  bp::class_<Solver, shared_ptr<Solver>, boost::noncopyable>(
    "Solver", bp::no_init)
    .add_property("net", &Solver::net)
    .add_property("test_nets", bp::make_function(&Solver::test_nets,
          bp::return_internal_reference<>()))
    .add_property("iter", &Solver::iter)
    .def("solve", static_cast<bool (Solver::*)(const char*)>(
          &Solver::Solve), SolveOverloads())
    .def("step", &Solver_StepNoGIL)
    .def("restore", &Solver::Restore)
    .def("snapshot", &Solver::Snapshot);
  BP_REGISTER_SHARED_PTR_TO_PYTHON(Solver);

  bp::class_<SGDSolver<Dtype>, bp::bases<Solver>,
    shared_ptr<SGDSolver<Dtype>>, boost::noncopyable>(
        "SGDSolver", bp::init<string>());
  bp::class_<NesterovSolver<Dtype>, bp::bases<Solver>,
    shared_ptr<NesterovSolver<Dtype> >, boost::noncopyable>(
        "NesterovSolver", bp::init<string>());
  bp::class_<AdaGradSolver<Dtype>, bp::bases<Solver>,
    shared_ptr<AdaGradSolver<Dtype> >, boost::noncopyable>(
        "AdaGradSolver", bp::init<string>());
  bp::class_<RMSPropSolver<Dtype>, bp::bases<Solver>,
    shared_ptr<RMSPropSolver<Dtype> >, boost::noncopyable>(
        "RMSPropSolver", bp::init<string>());
  bp::class_<AdaDeltaSolver<Dtype>, bp::bases<Solver>,
    shared_ptr<AdaDeltaSolver<Dtype> >, boost::noncopyable>(
        "AdaDeltaSolver", bp::init<string>());
  bp::class_<AdamSolver<Dtype>, bp::bases<Solver>,
    shared_ptr<AdamSolver<Dtype> >, boost::noncopyable>(
        "AdamSolver", bp::init<string>());

  bp::def("get_solver", &GetSolverFromFile,
      bp::return_value_policy<bp::manage_new_object>());

  // vector wrappers for all the vector types we use
  bp::class_<vector<shared_ptr<Blob> > >("BlobVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Blob> >, true>())
    .def("add_blob", bp::raw_function(&BlobVec_add_blob));

  bp::class_<vector<Blob*> >("RawBlobVec")
    .def(bp::vector_indexing_suite<vector<Blob*>, true>());
  bp::class_<vector<shared_ptr<LayerBase> > >("LayerBaseVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<LayerBase> >, true>());
  bp::class_<vector<shared_ptr<Layer<Dtype, Dtype> > > >("LayerVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Layer<Dtype, Dtype> > >, true>());
  bp::class_<vector<string> >("StringVec")
    .def(bp::vector_indexing_suite<vector<string> >());
  bp::class_<vector<int> >("IntVec")
    .def(bp::vector_indexing_suite<vector<int> >());
  bp::class_<vector<Dtype> >("DtypeVec")
    .def(bp::vector_indexing_suite<vector<Dtype> >());
  bp::class_<vector<shared_ptr<Net> > >("NetVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Net> >, true>());
  bp::class_<vector<bool> >("BoolVec")
    .def(bp::vector_indexing_suite<vector<bool> >());

  // boost python expects a void (missing) return value, while import_array
  // returns NULL for python3. import_array1() forces a void return value.
  import_array1();
}

}  // namespace caffe
