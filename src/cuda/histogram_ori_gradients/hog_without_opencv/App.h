/*
 * App.h
 *
 *  Created on: Sep 9, 2015
 *      Author: fernando
 */

#ifndef APP_H_
#define APP_H_

//new classes
#include "Args.h"

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <list>
#include <unistd.h>

#include "Helpful.h"
#ifdef LOGS
#include "log_helper.h"
#endif

using namespace std;
using namespace cv;

class App {
public:
    App(const Args& s);
    void run();

    void handleKey(char key);

    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;

    void workBegin();
    void workEnd();
    string workFps() const;

    string message() const;
    friend ostream &operator <<(ostream &os, const App &app);

private:
    App operator=(App&);

    Args args;
    bool running;

    bool use_gpu;
    bool make_gray;
    double scale;
    int gr_threshold;
    int nlevels;
    double hit_threshold;
    bool gamma_corr;

    int64 hog_work_begin;
    double hog_work_fps;

    int64 work_begin;
    double work_fps;
};

//only for debug
ostream &operator <<(ostream &os, const App &app) {
    os << "running " << app.running << endl;
    os << "make gray " << app.use_gpu << endl;
    os << "scale " << app.scale << endl;
    os << "gr threshold " << app.gr_threshold << endl;
    os << "nlevels " << app.nlevels << endl;
    os << "hit threshold " << app.hit_threshold << endl;
    os << "gamma corr " << app.gamma_corr << endl;
    return os;
}

App::App(const Args& s) {
    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

    args = s;
    use_gpu = true;
    make_gray = args.make_gray;
    scale = args.scale;
    gr_threshold = args.gr_threshold;
    nlevels = args.nlevels;

    if (args.hit_threshold_auto)
        args.hit_threshold = args.win_width == 48 ? 1.4 : 0.;
    hit_threshold = args.hit_threshold;

    gamma_corr = args.gamma_corr;

    if (args.win_width != 64 && args.win_width != 48)
        args.win_width = 64;
}


void App::run() {
    vector < vector<int> > gold;
    ifstream input_file(args.dst_video.c_str());
    //================== Init logs
#ifdef LOGS
    char test_info[90]; 
    vector<string> split_ = split(args.src, '/');
    int image_size = atoi(split_[split_.size() - 1].c_str());
    snprintf(test_info, 90, "gold %dx image", image_size);
    start_log_file("cudaHOG", test_info);
#endif
    //====================================
    if (!input_file.is_open()) {
#ifdef LOGS
        log_error_detail("Cant open gold file");
        end_log_file();
#endif
        throw runtime_error(string("can't open image file: " + args.dst_video));
    }

    //get gold file data
    string line;

    if (getline(input_file, line)) {
        vector < string > sep_line = split(line, ',');
        if (sep_line.size() != 7) {
#ifdef LOGS
            log_error_detail("wrong parameters on gold file");
            end_log_file();
#endif
            throw runtime_error(
                    string("wrong parameters on gold file: " + args.dst_video));
        }
        //all parameters are saved on gold generation on gold data, so it's just read from gold
        this->make_gray = (bool) atoi(sep_line[0].c_str());
        this->scale = atof(sep_line[1].c_str());
        this->gamma_corr = (bool) atoi(sep_line[2].c_str());
        this->gr_threshold = atoi(sep_line[3].c_str());
        args.win_width = atoi(sep_line[4].c_str());
        this->hit_threshold = atof(sep_line[5].c_str());
        this->nlevels = atoi(sep_line[6].c_str());
    }

    while (getline(input_file, line)) {
        vector < string > sep_line = split(line, ',');
        vector<int> values;
        for (int i = 0; i < GOLD_LINE_SIZE; i++) {
            values.push_back(atoi(sep_line[i].c_str()));
        }
        gold.push_back(values);
    }

    Size win_size(args.win_width, args.win_width * 2); //(64, 128) or (48, 96)
    Size win_stride(args.win_stride_width, args.win_stride_height);
    // Create HOG descriptors and detectors here
    vector<float> detector;
    if (win_size == Size(64, 128))
        detector = cv::gpu::HOGDescriptor::getPeopleDetector64x128();
    else
        detector = cv::gpu::HOGDescriptor::getPeopleDetector48x96();

    try {

        //gpu --------------------------------------------
        cv::gpu::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8),
                Size(8, 8), 9, cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2,
                gamma_corr, cv::gpu::HOGDescriptor::DEFAULT_NLEVELS);
        //====================================================
        //CPU ------------------------------------------------
        //cv::HOGDescriptor cpu_hog(win_size, Size(16, 16), Size(8, 8),
        //      Size(8, 8), 9, 1, -1, HOGDescriptor::L2Hys, 0.2, gamma_corr,
        //      cv::HOGDescriptor::DEFAULT_NLEVELS);
        //====================================================

        gpu_hog.setSVMDetector(detector);
        //cpu_hog.setSVMDetector(detector);
        for (int i = 0; i < args.iterations; i++) {
            Mat frame;

            frame = imread(args.src);
            if (frame.empty()) {
#ifdef LOGS
                log_error_detail("Cant open matrix Frame");
                end_log_file();
#endif
                throw runtime_error(
                        string("can't open image file: " + args.src));
            }

            Mat img_aux, img, img_to_show;
            gpu::GpuMat gpu_img;

            if (use_gpu)
                cvtColor(frame, img_aux, CV_BGR2BGRA);
            else
                frame.copyTo(img_aux);

            img = img_aux;
            img_to_show = img;

            gpu_hog.nlevels = nlevels;
            //cpu_hog.nlevels = nlevels;

            vector < Rect > found;

            // Perform HOG classification
            double time;
#ifdef LOGS
            start_iteration();
#endif
            gpu_img.upload(img);
            time = mysecond();
            gpu_hog.detectMultiScale(gpu_img, found, hit_threshold, win_stride,
                    Size(0, 0), scale, gr_threshold);
            time = mysecond() - time;
#ifdef LOGS
            end_iteration();
#endif
            cout << "Iteration: " << i << " Time: " << time << " ";
            //verify the output----------------------------------------------
            ostringstream error_detail;
            time = mysecond();


    //output verification
    //====================================
            unsigned long int error_counter = 0;
    //if the numbers of rects found is different from gold, log this info
            if(found.size() != gold.size()) {
                char message[120];
                snprintf(message, 120, "Rectangles found: %lu (gold has %lu).\n", found.size(), gold.size());
#ifdef LOGS
                log_error_detail(message);
#endif
            }
            for (size_t s = 0; s < found.size(); s++) {
                Rect r = found[s];
                vector<int> vf(GOLD_LINE_SIZE, 0);
                vf[0] = r.height;
                vf[1] = r.width;
                vf[2] = r.x;
                vf[3] = r.y;
                vf[4] = r.br().x;
                vf[5] = r.br().y;
                bool diff = set_countains(vf, gold);

                if (diff)
                    error_counter++;
            }
        //logs all found rectangles in case of any error
        if(error_counter){
                for(size_t g = 0; g < found.size(); g++){
                    Rect r = found[g];
                    char str[150];
                    snprintf(str, 150, "%d,%d,%d,%d,%d,%d\n", r.height, r.width, r.x,   r.y, r.br().x, r.br().y);
#ifdef LOGS
                    log_error_detail(str);
#endif
                }
            }
#ifdef LOGS
    // algorithm stops if 500+ errors in current iteration
            log_error_count(error_counter);
#endif
            
            cout << "Verification time " << mysecond() - time << endl;

        
            //stringstream ss;
            //ss << (i + 1);
            //imwrite(ss.str() + "_out.jpg", img_to_show);
        }
        //===============================================================
    } catch (cv::Exception &e) {
        char *str = const_cast<char*>(e.what());
        string err_msg(str);
#ifdef LOGS
        log_error_detail(str);
        end_log_file();
#endif
        throw runtime_error(string("error: " + err_msg));
    }
#ifdef LOGS
    end_log_file();
#endif
}

#endif /* APP_H_ */
