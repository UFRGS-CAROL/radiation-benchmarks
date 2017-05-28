/*
 * util.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <vector>
#include <cstdint>
#include <time.h>


#include <fstream>
#include <sstream>
#include <ostream>
#include <sys/time.h>


#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <math.h>
#include "boost/random.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

//#include "settings.h"
#pragma once
namespace convnet {


typedef float float_t;


typedef thrust::host_vector<float_t> vec_t;
typedef std::vector<std::vector<float_t> > vec2d_t;


inline int uniform_rand(int min, int max) {
    static boost::mt19937 gen(0);
    boost::uniform_smallint<> dst(min, max);
    return dst(gen);
}

template<typename T>
inline T uniform_rand(T min, T max) {
    static boost::mt19937 gen(0);
    boost::uniform_real<T> dst(min, max);
    return dst(gen);
}

template<typename Iter>
void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
    for (Iter it = begin; it != end; ++it)
        *it = uniform_rand(min, max);
}

void disp_vec_t(vec_t v) {
    for (auto i : v)
        std::cout << i << "\t";
    std::cout << "\n";
}

void disp_vec2d_t(vec2d_t v) {
    for (auto i : v) {
        for (auto i_ : i)
            std::cout << i_ << "\t";
        std::cout << "\n";
    }
}

float_t dot(vec_t x, vec_t w) {
    assert(x.size() == w.size());
    float_t sum = 0;
    for (size_t i = 0; i < x.size(); i++) {
        sum += x[i] * w[i];
    }
    return sum;
}

float_t dot_per_batch(int batch, vec_t x, vec_t w) {
    size_t x_width = w.size();
    float_t sum = 0;
    for (size_t i = 0; i < x_width; i++) {
        sum += x[batch * x_width + i] * w[i];
    }
    return sum;
}

struct Image {
    std::vector<std::vector<std::float_t> > img; // a image is represented by a 2-dimension vector
    size_t size; // width or height

    // construction
    Image(size_t size_, std::vector<std::vector<std::float_t> > img_) :
            img(img_), size(size_) {
    }

    // display the image
    void display() {
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                if (img[i][j] > 200)
                    std::cout << 1;
                else
                    std::cout << 0;
            }
            std::cout << std::endl;
        }
    }

    // up size to 32, make up with 0
    void upto_32() {
        assert(size < 32);

        std::vector<std::float_t> row(32, 0);

        for (size_t i = 0; i < size; i++) {
            img[i].insert(img[i].begin(), 0);
            img[i].insert(img[i].begin(), 0);
            img[i].push_back(0);
            img[i].push_back(0);
        }
        img.insert(img.begin(), row);
        img.insert(img.begin(), row);
        img.push_back(row);
        img.push_back(row);

        size = 32;
    }

    std::vector<std::float_t> extend() {
        std::vector<float_t> v;
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                v.push_back(img[i][j]);
            }
        }
        return v;
    }
};

typedef Image* Img;

struct Sample {
    uint8_t label; // label for a specific digit
    std::vector<float_t> image;
    Sample(float_t label_, std::vector<float_t> image_) :
            label(label_), image(image_) {
    }
};


} // namespace convnet


namespace jc {

std::string fileToString(const std::string& file_name) {
    std::string file_text;

    std::ifstream file_stream(file_name.c_str());
    if (!file_stream) {
        std::ostringstream oss;
        oss << "There is no file called " << file_name;
        throw std::runtime_error(oss.str());
    }

    file_text.assign(std::istreambuf_iterator<char>(file_stream), std::istreambuf_iterator<char>());

    return file_text;
}

unsigned int closestMultiple(unsigned int size, unsigned int divisor)
{
    unsigned int remainder = size % divisor;
    return remainder == 0 ? size : size - remainder + divisor;
}

template <class T>
void showMatrix(T *matrix, unsigned int width, unsigned int height)
{
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            std::cout << matrix[width*row + col] << " ";
        }
        std::cout << std::endl;
    }
    return;
}


class Timer {
    timeval start_;
    timeval stop_;

public:

    void start() {
        gettimeofday(&start_, NULL);
    }

    void stop() {
        gettimeofday(&stop_, NULL);
    }

    unsigned long getTime() const {
        unsigned long elapsed_time;
        elapsed_time = 1000 * 1000 * (stop_.tv_sec - start_.tv_sec);
        elapsed_time += (stop_.tv_usec - start_.tv_usec);
        return elapsed_time;
    }

    friend std::ostream& operator<<(std::ostream& oss, const Timer& t) {
        unsigned long factors[] = { 60000000, 1000000, 1000 };
        const char *time_scales[] = { "m", "s", "ms" };

        unsigned long rr = t.getTime();
        for (int i = 0; i < 3; ++i) {
            oss << rr / factors[i] << time_scales[i] << " ";
            rr %= factors[i];
        }
        oss << rr << "us";

        return oss;
    }

};




} //namespace jc

#endif /* UTIL_H_ */
