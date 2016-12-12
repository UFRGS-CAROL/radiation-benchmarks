// == mojo ====================================================================
//
//    Copyright (c) gnawice@gnawice.com. All rights reserved.
//	  See LICENSE in root folder
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files(the "Software"),
//    to deal in the Software without restriction, including without 
//    limitation the rights to use, copy, modify, merge, publish, distribute,
//    sublicense, and/or sell copies of the Software, and to permit persons to
//    whom the Software is furnished to do so, subject to the following 
//    conditions :
//
//    The above copyright notice and this permission notice shall be included
//    in all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
//    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
//    OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
//    THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ============================================================================
//    network.h: The main artificial neural network graph for mojo
// ==================================================================== mojo ==

#pragma once

#include <string>
#include <iostream> // cout
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

#include "layer.h"
#include "solver.h"
#include "activation.h"
#include "cost.h"

#ifdef LOGS
#include "../../../include/log_helper.h"
#endif

// hack for VS2010 to handle c++11 for(:)
#if (_MSC_VER  == 1600)
	#ifndef __for__
	#define __for__ for each
	#define __in__ in
	#endif
#else
	#ifndef __for__
	#define __for__ for
	#define __in__ :
	#endif
#endif

#ifdef MOJO_CV2
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#pragma comment(lib, "opencv_core249")
#pragma comment(lib, "opencv_highgui249")
#pragma comment(lib, "opencv_imgproc249")
#pragma comment(lib, "opencv_contrib249")
#endif

#ifdef MOJO_CV3
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#pragma comment(lib, "opencv_world310")
#endif

namespace mojo {

	// sleep needed for threading
#ifdef _WIN32
#include <windows.h>
	void mojo_sleep(unsigned milliseconds) { Sleep(milliseconds); }
#else
#include <unistd.h>
	void mojo_sleep(unsigned milliseconds) { usleep(milliseconds * 1000); }
#endif

	void replace_str(std::string& str, const std::string& from, const std::string& to) {
		if (from.empty())
			return;
		size_t start_pos = 0;
		while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
			str.replace(start_pos, from.length(), to);
			start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
		}
	}

#if defined(MOJO_CV2) || defined(MOJO_CV3)
	// transforms image. 
	// x_center, y_center of input
	// out dim is size of output w or h
	// theta in degrees
	cv::Mat matrix2cv(const mojo::matrix &m, bool uc8 = false)
	{
		cv::Mat cv_m;
		if (m.chans != 3)
		{
			cv_m = cv::Mat(m.cols, m.rows, CV_32FC1, m.x);
		}
		if (m.chans == 3)
		{
			cv::Mat in[3];
			in[0] = cv::Mat(m.cols, m.rows, CV_32FC1, m.x);
			in[1] = cv::Mat(m.cols, m.rows, CV_32FC1, &m.x[m.cols*m.rows]);
			in[2] = cv::Mat(m.cols, m.rows, CV_32FC1, &m.x[2 * m.cols*m.rows]);
			cv::merge(in, 3, cv_m);
		}
		if (uc8)
		{
			double min_, max_;
			cv_m = cv_m.reshape(1);
			cv::minMaxIdx(cv_m, &min_, &max_);
			cv_m = cv_m - min_;
			max_ = max_ - min_;
			cv_m /= max_;
			cv_m *= 255;
			cv_m = cv_m.reshape(m.chans, m.rows);
			if (m.chans != 3)
				cv_m.convertTo(cv_m, CV_8UC1);
			else
				cv_m.convertTo(cv_m, CV_8UC3);
		}
		return cv_m;
	}

	mojo::matrix cv2matrix(cv::Mat &m)
	{
		if (m.type() == CV_8UC1)
		{
			m.convertTo(m, CV_32FC1);
			m = m / 255.;
		}
		if (m.type() == CV_8UC3)
		{
			m.convertTo(m, CV_32FC3);
		}
		if (m.type() == CV_32FC1)
		{
			return mojo::matrix(m.cols, m.rows, 1, (float*)m.data);
		}
		if (m.type() == CV_32FC3)
		{
			cv::Mat in[3];
			cv::split(m, in);
			mojo::matrix out(m.cols, m.rows, 3);
			memcpy(out.x, in[0].data, m.cols*m.rows * sizeof(float));
			memcpy(&out.x[m.cols*m.rows], in[1].data, m.cols*m.rows * sizeof(float));
			memcpy(&out.x[2 * m.cols*m.rows], in[2].data, m.cols*m.rows * sizeof(float));
			return out;
		}
		return  mojo::matrix(0, 0, 0);
	}
	mojo::matrix transform(const mojo::matrix in, const int x_center, const int y_center,
		int out_dim, float theta = 0, float scale = 1.f)
	{
		const double _pi = 3.14159265358979323846;
		float cos_theta = (float)std::cos(theta / 180.*_pi);
		float sin_theta = (float)std::sin(theta / 180.*_pi);
		float half_dim = 0.5f*(float)out_dim / scale;

		cv::Point2f  pts1[4], pts2[4];
		pts1[0] = cv::Point2f(x_center - half_dim, y_center - half_dim);
		pts1[1] = cv::Point2f(x_center + half_dim, y_center - half_dim);
		pts1[2] = cv::Point2f(x_center + half_dim, y_center + half_dim);
		pts1[3] = cv::Point2f(x_center - half_dim, y_center + half_dim);

		pts2[0] = cv::Point2f(-half_dim, -half_dim);
		pts2[1] = cv::Point2f(half_dim, -half_dim);
		pts2[2] = cv::Point2f(half_dim, half_dim);
		pts2[3] = cv::Point2f(-half_dim, half_dim);

		// rotate around center spot
		for (int pt = 0; pt<4; pt++)
		{
			float x_t = (pts2[pt].x)*scale;
			float y_t = (pts2[pt].y)*scale;
			float x = cos_theta*x_t - sin_theta*y_t;
			float y = sin_theta*x_t + cos_theta*y_t;

			pts2[pt].x = x + (float)x_center;
			pts2[pt].y = y + (float)y_center;

			// we want to control how data is scaled down
			//		if (scale>1)
			//		{
			//			pts1[pt].x = pts1[pt].x / (float)scale;
			//			pts1[pt].y = pts1[pt].y / (float)scale;
			//		}
		}

		cv::Mat input = mojo::matrix2cv(in);

		//	if (scale>1)
		//		cv::resize(in, input, cv::Size(0, 0), 1. / scale, 1. / scale);
		//	else
		//		input = in;


		cv::Mat M = cv::getPerspectiveTransform(pts1, pts2);
		cv::Mat cv_out;

		cv::warpPerspective(input, cv_out,
			cv::getPerspectiveTransform(pts1, pts2),
			cv::Size((int)((float)out_dim), (int)((float)out_dim)),
			cv::INTER_AREA, cv::BORDER_REPLICATE); //cv::INTER_LINEAR

												   //INTER_AREA


												   //	double min;
												   //	cv::minMaxIdx(cv_out, &min);
												   //	std::cout << "min: " << min << "||";
		return mojo::cv2matrix(cv_out);
	}
#endif

// returns Energy (euclidian distance / 2) and max index
float match_labels(const float *out, const float *target, const int size, int *best_index = NULL)
{
	float E = 0;
	int max_j = 0;
	for (int j = 0; j<size; j++)
	{
		E += (out[j] - target[j])*(out[j] - target[j]);
		if (out[max_j]<out[j]) max_j = j;
	}
	if (best_index) *best_index = max_j;
	E *= 0.5;
	return E;
}
// returns index of highest value (argmax)
int max_index(const float *out, const int size)
{
	int max_j = 0;
	for (int j = 0; j<size; j++) if (out[max_j]<out[j]) max_j = j;
	return max_j;
}

//----------------------------------------------------------------------
//  network  
//  - class that holds all the layers and connection information
//	- runs forward prediction

class network
{
	
	int _size;  // output size
	int _thread_count; // determines number of layer sets (copys of layers)
	static const int MAIN_LAYER_SET = 0;

	// training related stuff
	int _batch_size;   // determines number of dW sets 
	float _skip_energy_level;
	bool _smart_train;
	std::vector <float> _running_E;
	double _running_sum_E;
	cost_function *_cost_function;
	solver *_solver;
	static const unsigned char BATCH_RESERVED = 1, BATCH_FREE = 0, BATCH_COMPLETE = 2;
	static const int BATCH_FILLED_COMPLETE = -2, BATCH_FILLED_IN_PROCESS = -1;
#ifdef MOJO_OMP
	omp_lock_t _lock_batch;
	void lock_batch() {omp_set_lock(&_lock_batch);}
	void unlock_batch() {omp_unset_lock(&_lock_batch);}
	void init_lock() {omp_init_lock(&_lock_batch);}
	void destroy_lock() {omp_destroy_lock(&_lock_batch);}
	int get_thread_num() {return omp_get_thread_num();}
#else
	void lock_batch() {}
	void unlock_batch() {}
	void init_lock(){}
	void destroy_lock() {}
	int get_thread_num() {return 0;}
#endif

public:	
	// training progress stuff
	int train_correct;
	int train_skipped;
	int stuck_counter;
	int train_updates;
	int train_samples;
	int epoch_count;
	int max_epochs;
	float best_estimated_accuracy;
	int best_accuracy_count;
	float old_estimated_accuracy;
	float estimated_accuracy;
// data augmentation stuff
	int use_augmentation; // 0=off, 1=mojo, 2=opencv
	int augment_x, augment_y;
	int augment_h_flip, augment_v_flip;
	mojo::pad_type augment_pad;
	float augment_theta;
	float augment_scale;



	// here we have multiple sets of the layers to allow threading and batch processing
	// a separate layer set is needed for each independent thread
	std::vector< std::vector<base_layer *>> layer_sets;
	
	std::map<std::string, int> layer_map;  // name-to-index of layer for layer management
	std::vector<std::pair<std::string, std::string>> layer_graph; // pairs of names of layers that are connected
	std::vector<matrix *> W; // these are the weights between/connecting layers 

	// these sets are needed because we need copies for each item in mini-batch
	std::vector< std::vector<matrix>> dW_sets; // only for training, will have _batch_size of these
	std::vector< std::vector<matrix>> dbias_sets; // only for training, will have _batch_size of these
	std::vector< unsigned char > batch_open; // only for training, will have _batch_size of these	
	

	network(const char* opt_name=NULL): _thread_count(1), _skip_energy_level(0.f), _batch_size(1) 
	{ 
		_size=0;  
		_solver = new_solver(opt_name);
		_cost_function = NULL;
		//std::vector<base_layer *> layer_set;
		//layer_sets.push_back(layer_set);
		layer_sets.resize(1);
		dW_sets.resize(_batch_size);
		dbias_sets.resize(_batch_size);
		batch_open.resize(_batch_size);
		_running_sum_E = 0.;
		train_correct = 0;
		train_samples = 0;
		train_skipped = 0;
		epoch_count = 0; 
		max_epochs = 1000;
		train_updates = 0;
		estimated_accuracy = 0;
		old_estimated_accuracy = 0;
		stuck_counter = 0;
		best_estimated_accuracy=0;
		best_accuracy_count=0;
		use_augmentation=0;
		augment_x = 0; augment_y = 0; augment_h_flip = 0; augment_v_flip = 0; 
		augment_pad =mojo::pad_type::edge; 
		augment_theta=0; augment_scale=0;

		init_lock();
#ifdef USE_AF
		af::setDevice(0);
        af::info();
#endif
	}
	
	~network() 
	{
		clear();
		if (_cost_function) delete _cost_function;
		if(_solver) delete _solver; 
		destroy_lock();	
	}

	// call clear if you want to load a different configuration/model
	void clear()
	{
		for(int i=0; i<(int)layer_sets.size(); i++)
		{
			__for__(auto l __in__ layer_sets[i]) delete l;
			layer_sets.clear();
		}
		layer_sets.clear();
		__for__(auto w __in__ W) if(w) delete w;  
		W.clear();
		layer_map.clear();
		layer_graph.clear();
	}

	// output size of final layer;
	int out_size() {return _size;}

	// get input size 
	bool get_input_size(int *w, int *h, int *c)
	{
		if(layer_sets[MAIN_LAYER_SET].size()<1) return false; 
		*w=layer_sets[MAIN_LAYER_SET][0]->node.cols;*h=layer_sets[MAIN_LAYER_SET][0]->node.rows;*c=layer_sets[MAIN_LAYER_SET][0]->node.chans;
		return true;
	}

	// sets up number of layer copies to run over multiple threads
	void build_layer_sets()
	{
		int layer_cnt = (int)layer_sets.size();
		if (layer_cnt<_thread_count) layer_sets.resize(_thread_count);
		// ToDo: add shrink back /  else if(layer_cnt>_thread_count)
		sync_layer_sets();
	}

	inline int get_thread_count() {return _thread_count;}
	// must call this with max thread count before constructing layers
	// value <1 will result in thread count = # cores (including hyperthreaded)
	void enable_omp(int threads = -1)
	{
#ifdef MOJO_OMP
		if (threads < 1) threads = omp_get_num_procs();
		omp_set_num_threads(threads);
#else
		if (threads < 1) _thread_count = 1;
		else _thread_count = threads;
		if (threads > 1) bail("must define MOJO_OMP to used threading");
#endif
		_thread_count = threads;
		build_layer_sets();
	}


	// when using threads, need to get bias data synched between all layer sets, 
	// call this after bias update in main layer set to copy the bias to the other sets
	void sync_layer_sets()
	{
		for(int i=1; i<(int)layer_sets.size();i++)
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				for(int k=0; k<layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++) 
					(layer_sets[i])[j]->bias.x[k]=(layer_sets[MAIN_LAYER_SET])[j]->bias.x[k];
	}

	// used to add some noise to weights
	void heat_weights()
	{
		__for__(auto w __in__ W)
		{
			if (!w) continue;
			matrix noise(w->cols, w->rows, w->chans);
			noise.fill_random_normal(1.f/ noise.size());
			//noise *= *w;
			*w += noise; 
		}
	}

	// used to add some noise to weights
	void remove_means()
	{
		__for__(auto w __in__ W)
			if(w) w->remove_mean();
	}

	// used to push a layer back in the ORDERED list of layers
	// if connect_all() is used, then the order of the push_back is used to connect the layers
	// when forward or backward propogation, this order is used for the serialized order of calculations 
	// Layer_name must be unique.
	bool push_back(const char *layer_name, const char *layer_config)
	{
		if(layer_map[layer_name]) return false; //already exists
		base_layer *l=new_layer(layer_name, layer_config);
		// set map to index

		// make sure there is a 'set' to add layers to
		if(layer_sets.size()<1)
		{
			std::vector<base_layer *> layer_set;
			layer_sets.push_back(layer_set);
		}
		// make sure layer_sets are created
		build_layer_sets();

		layer_map[layer_name] = (int)layer_sets[MAIN_LAYER_SET].size();
		layer_sets[MAIN_LAYER_SET].push_back(l);
		// upadate as potential last layer - so it sets the out size
		_size=l->fan_size();
		// add other copies needed for threading
		for(int i=1; i<(int)layer_sets.size();i++) layer_sets[i].push_back(new_layer(layer_name, layer_config));
		return true;
	}

	// connect 2 layers together and initialize weights
	// top and bottom concepts are reversed from literature
	// my 'top' is the input of a forward() pass and the 'bottom' is the output
	// perhaps 'top' traditionally comes from the brain model, but my 'top' comes
	// from reading order (information flows top to bottom)
	void connect(const char *layer_name_top, const char *layer_name_bottom) 
	{
		size_t i_top=layer_map[layer_name_top];
		size_t i_bottom=layer_map[layer_name_bottom];

		base_layer *l_top= layer_sets[MAIN_LAYER_SET][i_top];
		base_layer *l_bottom= layer_sets[MAIN_LAYER_SET][i_bottom];
		
		int w_i=(int)W.size();
		matrix *w = l_bottom->new_connection(*l_top, w_i);
		W.push_back(w);
		layer_graph.push_back(std::make_pair(layer_name_top,layer_name_bottom));
		// need to build connections for other batches/threads
		for(int i=1; i<(int)layer_sets.size(); i++)
		{
			l_top= layer_sets[i][i_top];
			l_bottom= layer_sets[i][i_bottom];
			delete l_bottom->new_connection(*l_top, w_i);
		}

		// we need to let solver prepare space for stateful information 
		if (_solver)
		{
			if (w)_solver->push_back(w->cols, w->rows, w->chans);
			else _solver->push_back(1, 1, 1);
		}

		int fan_in=l_bottom->fan_size();
		int fan_out=l_top->fan_size();

		// ToDo: this may be broke when 2 layers connect to one. need to fix (i.e. resnet)
		// after all connections, run through and do weights with correct fan count

		// initialize weights - ToDo: separate and allow users to configure(?)
		if (l_bottom->has_weights())
		{
			if (strcmp(l_bottom->p_act->name, "tanh") == 0)
			{
				// xavier : for tanh
				float weight_base = (float)(std::sqrt(6. / ((double)fan_in + (double)fan_out)));
				//		float weight_base = (float)(std::sqrt(.25/( (double)fan_in)));
				w->fill_random_uniform(weight_base);
			}
			else if ((strcmp(l_bottom->p_act->name, "sigmoid") == 0) || (strcmp(l_bottom->p_act->name, "sigmoid") == 0))
			{
				// xavier : for sigmoid
				float weight_base = 4.f*(float)(std::sqrt(6. / ((double)fan_in + (double)fan_out)));
				w->fill_random_uniform(weight_base);
			}
			else if ((strcmp(l_bottom->p_act->name, "lrelu") == 0) || (strcmp(l_bottom->p_act->name, "relu") == 0)
				|| (strcmp(l_bottom->p_act->name, "vlrelu") == 0) || (strcmp(l_bottom->p_act->name, "elu") == 0))
			{
				// he : for relu
				float weight_base = (float)(std::sqrt(2. / (double)fan_in));
				w->fill_random_normal(weight_base);
			}
			else
			{
				// lecun : orig
				float weight_base = (float)(std::sqrt(1. / (double)fan_in));
				w->fill_random_uniform(weight_base);
			}
		}
		else if (w) w->fill(0);
	}

	// automatically connect all layers in the order they were provided 
	// easy way to go, but can't deal with branch/highway/resnet/inception types of architectures
	void connect_all()
	{	
		for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size()-1; j++) 
			connect(layer_sets[MAIN_LAYER_SET][j]->name.c_str(), layer_sets[MAIN_LAYER_SET][j+1]->name.c_str());
	}

	// get the list of layers used (but not connection information)
	std::string get_configuration()
	{
		std::string str;
		// print all layer configs
		for (int j = 0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++) str+= "  "+layer_sets[MAIN_LAYER_SET][j]->name +" : " + layer_sets[MAIN_LAYER_SET][j]->get_config_string();
		str += "\n";
		// print layer links
		if (layer_graph.size() <= 0) return str;
		
		for (int j = 0; j < (int)layer_graph.size(); j++)
		{
			if (j % 3 == 0) str += "  ";
			if((j % 3 == 1)|| (j % 3 == 2)) str += ", ";
			str +=layer_graph[j].first + "-" + layer_graph[j].second;
			if (j % 3 == 2) str += "\n";
		}
		return str;
	}

	// performs forward pass and returns class index
	// do not delete or modify the returned pointer. it is a live pointer to the last layer in the network
	// if calling over multiple threads, provide the thread index since the interal data is not otherwise thread safe
	int predict_class(const float *in, int _thread_number = -1)
	{
		const float* out = forward(in, _thread_number);
		return max_index(out, out_size());
	}

	//----------------------------------------------------------------------------------------------------------
	// F O R W A R D
	//
	// the main forward pass 
	// if calling over multiple threads, provide the thread index since the interal data is not otherwise thread safe
	// train parameter is used to designate the forward pass is used in training (it turns on dropout layers, etc..)
	float* forward(const float *in, int _thread_number=-1, int _train=0)
	{
		if(_thread_number<0) _thread_number=get_thread_num();
		if (_thread_number > _thread_count) bail("need to call allow_threads()");
		if (_thread_number >= (int)layer_sets.size()) bail("need to call allow_threads()");

		// clear nodes to zero & find input layers
		std::vector<base_layer *> inputs;
		__for__(auto layer __in__ layer_sets[_thread_number])
		{
			if (dynamic_cast<input_layer*> (layer) != NULL)  inputs.push_back(layer);
			layer->node.fill(0.f);
		}
		// first layer assumed input. copy input to it 
		const float *in_ptr = in;
		//base_layer * layer = layer_sets[_thread_number][0];

		//memcpy(layer->node.x, in, sizeof(float)*layer->node.size());
		
		__for__(auto layer __in__ inputs)
		{
			memcpy(layer->node.x, in_ptr, sizeof(float)*layer->node.size());
			in_ptr += layer->node.size();
		}
		//for (int i = 0; i < layer->node.size(); i++)
		//	layer_sets[_thread_number][0]->node.x[i] = in[i];
		// for all layers
		__for__(auto layer __in__ layer_sets[_thread_number])
		{
			// add bias and activate these outputs (they should all be summed up from other branches at this point)
			layer->activate_nodes(); 

			// send output signal downstream (note in this code 'top' is input layer, 'bottom' is output - bucking tradition
			__for__ (auto &link __in__ layer->forward_linked_layers)
			{
				// instead of having a list of paired connections, just use the shape of W to determine connections
				// this is harder to read, but requires less look-ups
				// the 'link' variable is a std::pair created during the connect() call for the layers
				int connection_index = link.first; 
				base_layer *p_bottom = link.second;
				// weight distribution of the signal to layers under it
				p_bottom->accumulate_signal(*layer, *W[connection_index], _train);
			}

		}
		// return pointer to float * result from last layer
/*		std::cout << "out:";
		for (int i = 0; i < 10; i++)
		{
			std::cout << layer_sets[_thread_number][layer_sets[_thread_number].size() - 1]->node.x[i] <<",";
		}
		std::cout << "\n";
	*/
		return layer_sets[_thread_number][layer_sets[_thread_number].size()-1]->node.x;
	}


	// write parameters to stream/file
	// note that this does not persist intermediate training information that could be needed to 'pickup where you left off'
	bool write(std::ofstream& ofs, bool binary = false, bool final = false)
	{
		// save layers
		int layer_cnt = (int)layer_sets[MAIN_LAYER_SET].size();
//		int ignore_cnt = 0;
//		for (int j = 0; j<(int)layer_sets[0].size(); j++)
//			if (dynamic_cast<dropout_layer*> (layer_sets[0][j]) != NULL)  ignore_cnt++;
		ofs<<"mojo01" << std::endl;
		ofs<<(int)(layer_cnt)<<std::endl;
		
		for(int j=0; j<(int)layer_sets[0].size(); j++)
			ofs << layer_sets[MAIN_LAYER_SET][j]->name << std::endl << layer_sets[MAIN_LAYER_SET][j]->get_config_string();
//			if (dynamic_cast<dropout_layer*> (layer_sets[0][j]) != NULL)

		// save graph
		ofs<<(int)layer_graph.size()<<std::endl;
		for(int j=0; j<(int)layer_graph.size(); j++)
			ofs<<layer_graph[j].first << std::endl << layer_graph[j].second << std::endl;

		if(binary)
		{
			ofs<<(int)1<<std::endl;
			// binary version to save space if needed
			// save bias info
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				if(layer_sets[MAIN_LAYER_SET][j]->use_bias())
					ofs.write((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x, layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
			// save weights
			for (int j = 0; j < (int)W.size(); j++)
			{
				if (W[j])
					ofs.write((char*)W[j]->x, W[j]->size()*sizeof(float));
			}
		}
		else
		{
			ofs<<(int)0<<std::endl;
			// save bias info
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
			{
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
				{
					for (int k = 0; k < layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++)  ofs << layer_sets[MAIN_LAYER_SET][j]->bias.x[k] << " ";
					ofs << std::endl;
				}
			}
			// save weights
			for(int j=0; j<(int)W.size(); j++)
			{
				if (W[j])
				{
					for (int i = 0; i < W[j]->size(); i++) ofs << W[j]->x[i] << " ";
					ofs << std::endl;
				}
			}
		}
		ofs.flush();
		
		return true;
	}
	bool write(std::string &filename, bool binary = false, bool final = false) { 
		std::ofstream temp(filename.c_str(), std::ios::binary);
		return write(temp, binary, final);
	}//, std::ofstream::binary);

	// read network from a file/stream
	std::string getcleanline(std::istream &ifs)
	{
		std::string s;
		while (s.empty())
		{
			getline(ifs, s); // get version
			replace_str(s, "\r", "");
			if (ifs.eof()) break;
		}
		return s;
	}

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}
// trim from start (copying)
static inline std::string ltrimmed(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline std::string rtrimmed(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline std::string trimmed(std::string s) {
    trim(s);
    return s;
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

	int check_gold(char * filename)
	{
		std::ifstream ifs(filename,std::ios::binary);
		if(!ifs.good()) return 99999999;
		std::string s;
		s = getcleanline(ifs);
		int layer_count;
		int version = 0;
		if (s.compare("mojo01")==0)
		{
			ifs >> layer_count;
			ifs.ignore();
			version = 1;
		}

		int r_layer_count = (int)layer_sets[MAIN_LAYER_SET].size();

		int errors = 0;

		if(r_layer_count != layer_count){
                	errors++;
                	char error_detail[200];
                	sprintf(error_detail," layer_count, r: %d, e: %d", r_layer_count, layer_count);
#ifdef LOGS
                	log_error_detail(error_detail);
#endif
		}

		std::string layer_name;
		std::string layer_def;
		for (auto i=0; i<layer_count && i<r_layer_count; i++)
		{
			layer_name = getcleanline(ifs);
			trim(layer_name);
			if(layer_name.compare(trimmed(layer_sets[MAIN_LAYER_SET][i]->name)) != 0){
                		errors++;
                		char error_detail[200];
                		sprintf(error_detail," layer_name[%d], r: %s, e: %s", i, (trimmed(layer_sets[MAIN_LAYER_SET][i]->name)).c_str(), layer_name.c_str());
#ifdef LOGS
                		log_error_detail(error_detail);
#endif
			}
			layer_def = getcleanline(ifs);
			trim(layer_def);
			if(layer_def.compare(trimmed(layer_sets[MAIN_LAYER_SET][i]->get_config_string())) != 0){
                		errors++;
                		char error_detail[200];
                		sprintf(error_detail," layer_name[%d], r: %s, e: %s", i, (trimmed(layer_sets[MAIN_LAYER_SET][i]->get_config_string())).c_str(), layer_def.c_str());
#ifdef LOGS
                		log_error_detail(error_detail);
#endif
			}
		}

		// read graph
		int graph_count;
		ifs>>graph_count;
		getline(ifs,s); // get endline
		int r_graph_count = (int)layer_graph.size();
		if(r_graph_count != graph_count){
                	errors++;
                	char error_detail[200];
                	sprintf(error_detail," graph_count, r: %d, e: %d", r_graph_count, graph_count);
#ifdef LOGS
                	log_error_detail(error_detail);
#endif
		}

		std::string layer_name1;
		std::string layer_name2;
		for (auto i=0; i<graph_count && i<r_graph_count; i++)
		{
			layer_name1= getcleanline(ifs);
			trim(layer_name1);
			if(layer_name1.compare(layer_graph[i].first) != 0){
                		errors++;
                		char error_detail[200];
                		sprintf(error_detail," graph_name1[%d], r: %s, e: %s", i, (layer_graph[i].first).c_str(), layer_name1.c_str());
#ifdef LOGS
                		log_error_detail(error_detail);
#endif
			}
			layer_name2 = getcleanline(ifs);
			trim(layer_name2);
			if(layer_name2.compare(layer_graph[i].second) != 0){
                		errors++;
                		char error_detail[200];
                		sprintf(error_detail," graph_name2[%d], r: %s, e: %s", i, (layer_graph[i].second).c_str(), layer_name2.c_str());
#ifdef LOGS
                		log_error_detail(error_detail);
#endif
			}
		}

		int binary;
		ifs>>binary;
		getline(ifs,s); // get endline

		// binary version to save space if needed
		if(binary==1)
		{
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias()){
					float * bias = (float *)malloc(sizeof(float)*layer_sets[MAIN_LAYER_SET][j]->bias.size());
					float * read = layer_sets[MAIN_LAYER_SET][j]->bias.x;
					ifs.read((char*)bias, 
						layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
					for(int i = 0; i < layer_sets[MAIN_LAYER_SET][j]->bias.size(); i++){
                				if ((fabs((read[i] - bias[i]) / read[i]) > 0.0000000001) || (fabs((read[i] - bias[i]) / bias[i]) > 0.0000000001)) {
                				    errors++;
#ifdef LOGS
                				    char error_detail[200];
                				    sprintf(error_detail," bias: [%d], r: %1.16e, e: %1.16e", i, read[i], bias[i]);
                				    log_error_detail(error_detail);
#endif
                				}
					
					}
					free(bias);
				}
			for (int j = 0; j < (int)W.size(); j++)
			{

				if (W[j])
				{
					float * weight = (float *)malloc(sizeof(float)*W[j]->size());
					float * read = W[j]->x;
					ifs.read((char*)weight, W[j]->size()*sizeof(float));
					for(int i = 0; i < W[j]->size(); i++){
                				if ((fabs((read[i] - weight[i]) / read[i]) > 0.0000000001) || (fabs((read[i] - weight[i]) / weight[i]) > 0.0000000001)) {
                				    errors++;
#ifdef LOGS
                				    char error_detail[200];
                				    sprintf(error_detail," weight: [%d], r: %1.16e, e: %1.16e", i, read[i], weight[i]);
                				    log_error_detail(error_detail);
#endif
                				}
					
					}
					free(weight);
				}
			}
		}

#ifdef LOGS
        	log_error_count(errors);
#endif
		return errors;
	}
	
	bool read(std::istream &ifs)
	{
		if(!ifs.good()) return false;
		std::string s;
		s = getcleanline(ifs);
		int layer_count;
		int version = 0;
		if (s.compare("mojo01")==0)
		{
			ifs >> layer_count;
			ifs.ignore();
			//getline(ifs, s); // get endline
			version = 1;
		}
		else if (s.compare("mojo:") == 0)
		{
			version = -1;
			int cnt = 1;

			while (!ifs.eof())
			{
				s = getcleanline(ifs);
				if (s.empty()) continue;
				push_back(int2str(cnt).c_str(), s.c_str());
				cnt++;
			}
			connect_all();

			// copies batch=0 stuff to other batches
			sync_layer_sets();
			return true;
		}
		else
			layer_count = atoi(s.c_str());

		// read layer def
		//ifs>>layer_count;

		std::string layer_name;
		std::string layer_def;
		for (auto i=0; i<layer_count; i++)
		{
			layer_name = getcleanline(ifs);
			//getline(ifs,layer_name);
			//replace_str(layer_name, "\r", "");
			//getline(ifs,layer_def);
			layer_def = getcleanline(ifs);
			push_back(layer_name.c_str(),layer_def.c_str());
		}

		// read graph
		int graph_count;
		ifs>>graph_count;
		getline(ifs,s); // get endline

		std::string layer_name1;
		std::string layer_name2;
		for (auto i=0; i<graph_count; i++)
		{
			layer_name1= getcleanline(ifs);
			layer_name2 = getcleanline(ifs);
			connect(layer_name1.c_str(),layer_name2.c_str());
		}

		int binary;
		ifs>>binary;
		getline(ifs,s); // get endline

		// binary version to save space if needed
		if(binary==1)
		{
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
					ifs.read((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x, 
						layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
			for (int j = 0; j < (int)W.size(); j++)
			{

				if (W[j])
				{
					ifs.read((char*)W[j]->x, W[j]->size()*sizeof(float));
				}
			}
		}
		else if(binary==0)// text version
		{
			// read bias
			for(int j=0; j<layer_count; j++)
			{
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
				{
					for (int k = 0; k < layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++)  ifs >> layer_sets[MAIN_LAYER_SET][j]->bias.x[k];
					ifs.ignore();// getline(ifs, s); // get endline
				}
			}

			// read weights
			for (auto j=0; j<(int)W.size(); j++)
			{
				if (W[j])
				{
					for (int i = 0; i < W[j]->size(); i++) ifs >> W[j]->x[i];
					ifs.ignore(); //getline(ifs, s); // get endline
				}
			}
		}
	
		// copies batch=0 stuff to other batches
		sync_layer_sets();

		return true;
	}
	bool read(std::string filename)
	{
		std::ifstream fs(filename.c_str(),std::ios::binary);
		if (fs.is_open())
		{
			bool ret = read(fs);
			fs.close();
			return ret;
		}
		else return false;
	}
	bool read(const char *filename) { return  read(std::string(filename)); }

#ifndef NO_TRAINING_CODE  // this is surely broke by now and will need to be fixed

	// resets the state of all batches to 'free' state
	void reset_mini_batch() { memset(batch_open.data(), BATCH_FREE, batch_open.size()); }
	
	// sets up number of mini batches (storage for sets of weight deltas)
	void set_mini_batch_size(int batch_cnt)
	{
		if (batch_cnt<1) batch_cnt = 1;
		_batch_size = batch_cnt;
		dW_sets.resize(_batch_size);
		dbias_sets.resize(_batch_size);
		batch_open.resize(_batch_size); 
		reset_mini_batch();
	}
	
	int get_mini_batch_size() { return _batch_size; }

	// return index of next free batch
	// or returns -2 (BATCH_FILLED_COMPLETE) if no free batches - all complete (need a sync call)
	// or returns -1 (BATCH_FILLED_IN_PROCESS) if no free batches - some still in progress (must wait to see if one frees)
	int get_next_open_batch()
	{
		int reserved = 0;
		int filled = 0;
		for (int i = 0; i<batch_open.size(); i++)
		{
			if (batch_open[i] == BATCH_FREE) return i;
			if (batch_open[i] == BATCH_RESERVED) reserved++;
			if (batch_open[i] == BATCH_COMPLETE) filled++;
		}
		if (reserved>0) return BATCH_FILLED_IN_PROCESS; // all filled but wainting for reserves
		if (filled == batch_open.size()) return BATCH_FILLED_COMPLETE; // all filled and complete
		
		bail("threading error"); // should not get here  unless threading problem
	}

	// apply all weights to first set of dW, then apply to model weights 
	void sync_mini_batch()
	{
		// need to ensure no batches in progress (reserved)
		int next = get_next_open_batch();
		if (next == BATCH_FILLED_IN_PROCESS) bail("thread lock");

		int layer_cnt = (int)layer_sets[MAIN_LAYER_SET].size();

		base_layer *layer;

		 // sum contributions 
		for (int k = layer_cnt - 1; k >= 0; k--)
		{
			layer = layer_sets[MAIN_LAYER_SET][k];
			__for__(auto &link __in__ layer->backward_linked_layers)
			{
				int w_index = (int)link.first;
				// if batch free, then make sure it is zero'd out because we will increment dW set [0]
				if (batch_open[0] == BATCH_FREE) dW_sets[0][w_index].fill(0);
				for (int b = 1; b< _batch_size; b++)
				{
					if (batch_open[b] == BATCH_COMPLETE) dW_sets[0][w_index] += dW_sets[b][w_index];
				}
			}
			if (dynamic_cast<convolution_layer*> (layer) != NULL)  continue;

			// bias stuff... that needs to be fixed for conv layers perhaps
			if (batch_open[0] == BATCH_FREE) dbias_sets[0][k].fill(0);
			for (int b = 1; b< _batch_size; b++)
			{
				if (batch_open[b] == BATCH_COMPLETE) dbias_sets[0][k] += dbias_sets[b][k];
			}
		}

		// update weights
		for (int k = layer_cnt - 1; k >= 0; k--)
		{
			layer = layer_sets[MAIN_LAYER_SET][k];
			__for__(auto &link __in__ layer->backward_linked_layers)
			{
				int w_index = (int)link.first;
				if (dW_sets[MAIN_LAYER_SET][w_index].size() > 0)
					if(W[w_index]) _solver->increment_w(W[w_index], w_index, dW_sets[MAIN_LAYER_SET][w_index]);  // -- 10%

			}
			layer->update_bias(dbias_sets[0][k], _solver->learning_rate);
		}

	
		// prepare to start mini batch over
		reset_mini_batch();
		train_updates++; // could have no updates .. so this is not exact
		sync_layer_sets();

	}

	// reserve_next.. is used to reserve a space in the minibatch for the existing training sample
	int reserve_next_batch()
	{
		lock_batch();
		int my_batch_index = -3;
		while (my_batch_index < 0)
		{
			my_batch_index = get_next_open_batch();

			if (my_batch_index >= 0) // valid index
			{
				batch_open[my_batch_index] = BATCH_RESERVED;
				unlock_batch();
				return my_batch_index;
			}
			else if (my_batch_index == BATCH_FILLED_COMPLETE) // all index are complete
			{
				sync_mini_batch(); // resets _batch_index to 0
				my_batch_index = get_next_open_batch();
				batch_open[my_batch_index] = BATCH_RESERVED;
				unlock_batch();
				return my_batch_index;
			}
			// need to wait for ones in progress to finish
			unlock_batch();
			mojo_sleep(1);
			lock_batch();
		}
		return -3;
	}

	float get_learning_rate() {if(!_solver) bail("set solver"); return _solver->learning_rate;}
	void set_learning_rate(float alpha) {if(!_solver) bail("set solver"); _solver->learning_rate=alpha;}
	void reset_solver() {if(!_solver) bail("set solver"); _solver->reset();}
	bool get_smart_training() {return _smart_train;}
	void set_smart_training(bool _use_train) { _smart_train = _use_train;}
	float get_smart_train_level() { return _skip_energy_level; }
	void set_smart_train_level(float _level) { _skip_energy_level = _level; }
	void set_max_epochs(int max_e) { if (max_e <= 0) max_e = 1; max_epochs = max_e; }
	int get_epoch() { return epoch_count; }

	// goal here is to update the weights W. 
	// use w_new = w_old - alpha dE/dw
	// E = sum: 1/2*||y-target||^2
	// note y = f(x*w)
	// dE = (target-y)*dy/dw = (target-y)*df/dw = (target-y)*df/dx* dx/dw = (target-y) * df * y_prev  
	// similarly for cross entropy

// ===========================================================================
// training part
// ===========================================================================

	void set_random_augmentation(int translate_x, int translate_y,
		int flip_h, int flip_v, mojo::pad_type padding = mojo::pad_type::edge)
	{
		use_augmentation = 1;
		augment_x = translate_x;
		augment_y = translate_y;
		augment_h_flip = flip_h;
		augment_v_flip = flip_v;
		augment_pad = padding;
		augment_theta = 0;
		augment_scale = 0;

	}
	void set_random_augmentation(int translate_x, int translate_y,
		int flip_h, int flip_v, float rotation_deg, float scale, mojo::pad_type padding = mojo::pad_type::edge)
	{
		use_augmentation = 2;
		augment_x = translate_x;
		augment_y = translate_y;
		augment_h_flip = flip_h;
		augment_v_flip = flip_v;
		augment_pad = padding;
		augment_theta = rotation_deg;
		augment_scale = scale;

	}

	// call before starting training for current epoch
	void start_epoch(std::string loss_function="mse")
	{
		_cost_function=new_cost_function(loss_function);
		train_correct = 0;
		train_skipped = 0;
		train_updates = 0;
		train_samples = 0;
		if (epoch_count == 0) reset_solver();
	
		// accuracy not improving .. slow learning
		if(_smart_train &&  (best_accuracy_count > 4))
		{
			stuck_counter++;
			set_learning_rate((0.5f)*get_learning_rate());
			if (get_learning_rate() < 0.000001f)
			{
//				heat_weights();
				set_learning_rate(0.000001f);
				stuck_counter++;// end of the line.. so speed up end
			}
			best_accuracy_count = 0;
		}

		old_estimated_accuracy = estimated_accuracy;
		estimated_accuracy = 0;
		//_skip_energy_level = 0.05;
		_running_sum_E = 0;
	}
	
	// time to stop?
	bool elvis_left_the_building()
	{
		// 2 stuck x 4 non best accuracy to quit = 8 times no improvement 
		if ((epoch_count>max_epochs) || (stuck_counter > 3)) return true;
		else return false;
	}

	// call after putting all training samples through this epoch
	bool end_epoch()
	{
		// run leftovers through mini-batch
		sync_mini_batch();
		epoch_count++;

		// estimate accuracy of validation run 
		estimated_accuracy = 100.f*train_correct / train_samples;

		if (train_correct > best_estimated_accuracy)
		{
			best_estimated_accuracy = (float)train_correct;
			best_accuracy_count = 0;
			stuck_counter = 0;
		}
		else best_accuracy_count++;

		return elvis_left_the_building();
	}

	// if smart training was thinking about exiting, calling reset will make it think everything is OK
	void reset_smart_training()
	{
		stuck_counter=0;
		best_accuracy_count = 0;
		best_estimated_accuracy = 0;
	}
	void update_smart_train(const float E, bool correct)
	{

#ifdef MOJO_OMP	
#pragma omp critical
#endif
		{
			train_samples++;
			if (correct) train_correct++;

			if (_smart_train)
			{
				_running_E.push_back(E);
				_running_sum_E += E;
				const int SMART_TRAIN_SAMPLE_SIZE = 1000;

				int s = (int)_running_E.size();
				if (s >= SMART_TRAIN_SAMPLE_SIZE)
				{
					_running_sum_E /= (double)s;
					std::sort(_running_E.begin(), _running_E.end());
					float top_fraction = (float)_running_sum_E*10.f; //10.
					const float max_fraction = 0.75f;
					const float min_fraction = 0.075f;// 0.03f;

					if (top_fraction > max_fraction) top_fraction = max_fraction;
					if (top_fraction < min_fraction) top_fraction = min_fraction;
					int index = s - 1 - (int)(top_fraction*(s - 1));

					if (_running_E[index] > 0) _skip_energy_level = _running_E[index];

					_running_sum_E = 0;

					_running_E.clear();
				}
			}
			if (E > 0 && E < _skip_energy_level)
			{
				//std::cout << "E=" << E;
				train_skipped++;
			}

		}  // omp critical


	}
	// finish back propogation through the hidden layers
	void backward_hidden(const int my_batch_index, const int thread_number)
	{
		const int layer_cnt = (int)layer_sets[thread_number].size();
		const int last_layer_index = layer_cnt - 1;
		base_layer *layer;// = layer_sets[thread_number][last_layer_index];

		// update hidden layers
		// start at lower layer and push information up to previous layer
		// handle dropout first

		for (int k = last_layer_index; k >= 0; k--)
		{
			layer = layer_sets[thread_number][k];
			// all the signals should be summed up to this layer by now, so we go through and take the grad of activiation
			int nodes = layer->node.size();
			// already did last layer, so skip it
			if (k< last_layer_index)
				for (int i = 0; i< nodes; i++)
					layer->delta.x[i] *= layer->df(layer->node.x, i, nodes);

			// now pass that signal upstream
			__for__(auto &link __in__ layer->backward_linked_layers) // --- 50% of time this loop
			{
				base_layer *p_top = link.second;
				// note all the delta[connections[i].second] should have been calculated by time we get here
				layer->distribute_delta(*p_top, *W[link.first]);
			}
		}


		// update weights - shouldn't matter the direction we update these 
		// we can stay in backwards direction...
		// it was not faster to combine distribute_delta and increment_w into the same loop
		int size_W = (int)W.size();
		dW_sets[my_batch_index].resize(size_W);
		dbias_sets[my_batch_index].resize(layer_cnt);
		for (int k = last_layer_index; k >= 0; k--)
		{
			layer = layer_sets[thread_number][k];

			__for__(auto &link __in__ layer->backward_linked_layers)
			{
				base_layer *p_top = link.second;
				int w_index = (int)link.first;
				//if (dynamic_cast<max_pooling_layer*> (layer) != NULL)  continue;
				layer->calculate_dw(*p_top, dW_sets[my_batch_index][w_index]);// --- 20%
																			  // moved this out to sync_mini_batch();
																			  //_solver->increment_w( W[w_index],w_index, dW_sets[_batch_index][w_index]);  // -- 10%
			}
			if (dynamic_cast<convolution_layer*> (layer) != NULL)  continue;

			dbias_sets[my_batch_index][k] = layer->delta;
		}
		// if all batches finished, update weights
		lock_batch();
		batch_open[my_batch_index] = BATCH_COMPLETE;
		int next_index = get_next_open_batch();
		if (next_index == BATCH_FILLED_COMPLETE) // all complete
			sync_mini_batch(); // resets _batch_index to 0
		unlock_batch();
	}

	// after starting epoch, call this to train against a class label
	// label_index must be 0 to out_size()-1
	// for thread safety, you must pass in the thread_index if calling from different threads

	bool train_class(float *in, int label_index, int _thread_number = -1)
	{
		if (_solver == NULL) bail("set solver");
		if (_thread_number < 0) _thread_number = get_thread_num();
		if (_thread_number > _thread_count)  bail("call allow_threads()");

		const int thread_number = _thread_number;

		float *input = in;
		mojo::matrix augmented_input;
		if (use_augmentation > 0)
		{
			//augment_h_flip = flip_h;
			//augment_v_flip = flip_v;
			// copy input to matrix type
			mojo::matrix m(layer_sets[thread_number][0]->node.cols, layer_sets[thread_number][0]->node.rows, layer_sets[thread_number][0]->node.chans, in);
#if defined(MOJO_CV2) || defined(MOJO_CV3)
			if (augment_theta > 0 || augment_scale > 0)
			{
				float s = ((float)(rand() % 101) / 50.f - 1.f)*augment_scale;
				float t = ((float)(rand() % 101) / 50.f - 1.f)*augment_theta;
				m = transform(m, m.cols / 2, m.rows / 2, m.cols, t, 1+s);
			}
#endif
			if (augment_h_flip)
				if ((rand() % 2) == 0)
					m = m.flip_cols();
			if (augment_v_flip)
				if ((rand() % 2) == 0)
					m = m.flip_rows();
			augmented_input = m.shift((rand() % (augment_x * 2 + 1)) - augment_x, (rand() % (augment_y * 2 + 1)) - augment_y, augment_pad);
			
			input = augmented_input.x;
		}
	



		// get next free mini_batch slot
		// this is tied to the current state of the model
		int my_batch_index = reserve_next_batch();
		// out of data or an error if index is negative
		if (my_batch_index < 0) return false;
		// run through forward to get nodes activated
		forward(input, thread_number, 1);

		// set all deltas to zero
		__for__(auto layer __in__ layer_sets[thread_number]) layer->delta.fill(0.f);

		int layer_cnt = (int)layer_sets[thread_number].size();

		// calc delta for last layer to prop back up through network
		// d = (target-out)* grad_activiation(out)
		const int last_layer_index = layer_cnt - 1;
		base_layer *layer = layer_sets[thread_number][last_layer_index];
		const int layer_node_size = layer->node.size();

		if (dynamic_cast<dropout_layer*> (layer) != NULL) bail("can't have dropout on last layer");

		float E = 0;
		int max_j_out = 0;
		int max_j_target = label_index;

		// was passing this in, but may as well just create it on the fly
		// a vector mapping the label index to the desired target output node values
		// all -1 except target node 1
		std::vector<float> target;
		if((std::string("sigmoid").compare(layer->p_act->name) == 0)
			|| (std::string("softmax").compare(layer->p_act->name) == 0))
			target = std::vector<float>(layer_node_size, 0);
		else
			target = std::vector<float>(layer_node_size, -1);
		if(label_index>=0 && label_index<layer_node_size) target[label_index] = 1;

		const float grad_fudge = 1.0f;
		// because of numerator/demoninator cancellations which prevent a divide by zero issue, 
		// we need to handle some things special on output layer
		float cost_activation_type = 0;
		if ((std::string("sigmoid").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0)) 
			cost_activation_type = 1;
		else if ((std::string("softmax").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0))
			cost_activation_type = 1;
		else if ((std::string("tanh").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0)) 
			cost_activation_type = 4;
	
		for (int j = 0; j < layer_node_size; j++)
		{
			if(cost_activation_type>0)
				layer->delta.x[j] = grad_fudge*cost_activation_type*(layer->node.x[j]- target[j]);
			else
				layer->delta.x[j] = grad_fudge*_cost_function->d_cost(layer->node.x[j], target[j])*layer->df(layer->node.x, j, layer_node_size);

			// pick best response
			if (layer->node.x[max_j_out] < layer->node.x[j]) max_j_out = j;
			// for better E maybe just look at 2 highest scores so zeros don't dominate 

			E += mse::cost(layer->node.x[j], target[j]);
		}
	
		E /= (float)layer_node_size;
		// check for NAN
		if (E != E) bail("network blew up - try lowering learning rate\n");
		
		// critical section in here, blocking update
		bool match = false;
		if ((max_j_target == max_j_out)) match = true;
		update_smart_train(E, match);

		if (E>0 && E<_skip_energy_level && _smart_train && match)
		{
			lock_batch();
			batch_open[my_batch_index] = BATCH_FREE;
			unlock_batch();
			return false;  // return without doing training
		}
		backward_hidden(my_batch_index, thread_number);
		return true;
	}
	
	// after starting epoch, call this to train against a target vector
	// for thread safety, you must pass in the thread_index if calling from different threads
	// if positive=1, goal is to minimize the distance between in and target
	bool train_target(float *in, float *target, int positive=1, int _thread_number = -1)
	{
		if (_solver == NULL) bail("set solver");
		if (_thread_number < 0) _thread_number = get_thread_num();
		if (_thread_number > _thread_count)  bail("call allow_threads()");

		const int thread_number = _thread_number;

		// get next free mini_batch slot
		// this is tied to the current state of the model
		int my_batch_index = reserve_next_batch();
		// out of data or an error if index is negative
		if (my_batch_index < 0) return false;
		// run through forward to get nodes activated
		float *out=forward(in, thread_number, 1);

		// set all deltas to zero
		__for__(auto layer __in__ layer_sets[thread_number]) layer->delta.fill(0.f);

		int layer_cnt = (int)layer_sets[thread_number].size();

		// calc delta for last layer to prop back up through network
		// d = (target-out)* grad_activiation(out)
		const int last_layer_index = layer_cnt - 1;
		base_layer *layer = layer_sets[thread_number][last_layer_index];
		const int layer_node_size = layer->node.size();

		if (dynamic_cast<dropout_layer*> (layer) != NULL) bail("can't have dropout on last layer");

		float E = 0;
		int max_j_out = 0;
		//int max_j_target = label_index;

		// was passing this in, but may as well just create it on the fly
		// a vector mapping the label index to the desired target output node values
		// all -1 except target node 1
//		std::vector<float> target;
//		if ((std::string("sigmoid").compare(layer->p_act->name) == 0)
//			|| (std::string("softmax").compare(layer->p_act->name) == 0))
//			target = std::vector<float>(layer_node_size, 0);
//		else
//			target = std::vector<float>(layer_node_size, -1);
//		if (label_index >= 0 && label_index<layer_node_size) target[label_index] = 1;

		const float grad_fudge = 1.0f;
		// because of numerator/demoninator cancellations which prevent a divide by zero issue, 
		// we need to handle some things special on output layer
		float cost_activation_type = 0;
		if ((std::string("sigmoid").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0))
			cost_activation_type = 1;
		else if ((std::string("softmax").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0))
			cost_activation_type = 1;
		else if ((std::string("tanh").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0))
			cost_activation_type = 4;

		for (int j = 0; j < layer_node_size; j++)
		{
			if (positive) // want to minimize distance
			{
				if (cost_activation_type > 0)
					layer->delta.x[j] = grad_fudge*cost_activation_type*(layer->node.x[j] - target[j]);
				else
					layer->delta.x[j] = grad_fudge*_cost_function->d_cost(layer->node.x[j], target[j])*layer->df(layer->node.x, j, layer_node_size);
			}
			else
			{
				if (cost_activation_type > 0)
					layer->delta.x[j] = grad_fudge*cost_activation_type*(1.f-abs(layer->node.x[j] - target[j]));
				else
					layer->delta.x[j] = grad_fudge*(1.f-abs(_cost_function->d_cost(layer->node.x[j], target[j])))*layer->df(layer->node.x, j, layer_node_size);
			}
			// pick best response
			if (layer->node.x[max_j_out] < layer->node.x[j]) max_j_out = j;
			// for better E maybe just look at 2 highest scores so zeros don't dominate 

			// L2 distance x 2
			E += mse::cost(layer->node.x[j], target[j]);
		}

		E /= (float)layer_node_size;
		// check for NAN
		if (E != E) bail("network blew up - try lowering learning rate\n");

		// critical section in here, blocking update
		bool match = false;
// FIxME		if ((max_j_target == max_j_out)) match = true;
		update_smart_train(E, match);

		if (E>0 && E<_skip_energy_level && _smart_train && match)
		{
			lock_batch();
			batch_open[my_batch_index] = BATCH_FREE;
			unlock_batch();
			return false;  // return without doing training
		}
		backward_hidden(my_batch_index, thread_number);
		return true;
	}

#else

	float get_learning_rate() {return 0;}
	void set_learning_rate(float alpha) {}
	void train(float *in, float *target){}
	void reset() {}
	float get_smart_train_level() {return 0;}
	void set_smart_train_level(float _level) {}
	bool get_smart_train() { return false; }
	void set_smart_train(bool _use) {}

#endif

};

}
