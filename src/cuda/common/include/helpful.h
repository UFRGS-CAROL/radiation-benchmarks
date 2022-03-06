/*
 * Helpful.h
 *
 *  Created on: Sep 17, 2015
 *      Author: fernando
 */

#ifndef HELPFUL_H_
#define HELPFUL_H_
#include <sys/time.h>

#define GOLD_LINE_SIZE 6

#ifdef __cplusplus

#include <sstream>
//~ #include <iostream>
#include <fstream>

static void dump_output(int iteration_num, std::string directory, bool corrupted,
        std::vector<std::vector<int> > data) {
    char filename[100];

    time_t file_time;
    struct tm *ptm;
    char day[10], month[10], year[15], hour[10], second[10], minute[10];
    char str_file_time[80] = "";

    file_time = time(NULL);
    ptm = gmtime(&file_time);

    snprintf(day, sizeof(day), "%02d", ptm->tm_mday);
    snprintf(month, sizeof(month), "%02d", ptm->tm_mon + 1);
    snprintf(year, sizeof(year), "%04d", ptm->tm_year + 1900);
    snprintf(hour, sizeof(hour), "%02d", ptm->tm_hour);
    snprintf(minute, sizeof(minute), "%02d", ptm->tm_min);
    snprintf(second, sizeof(second), "%02d", ptm->tm_sec);

    strcpy(str_file_time, year);
    strcat(str_file_time, "_");
    strcat(str_file_time, month);
    strcat(str_file_time, "_");
    strcat(str_file_time, day);
    strcat(str_file_time, "_");

    strcat(str_file_time, hour);
    strcat(str_file_time, "_");
    strcat(str_file_time, minute);
    strcat(str_file_time, "_");
    strcat(str_file_time, second);
    strcat(str_file_time, "_");

    if (corrupted)
        sprintf(filename, "%s/graph%05d_corrupted_%s.data", directory.c_str(),
                iteration_num, str_file_time);
    else
        sprintf(filename, "%s/graph%05d_%s.data", directory.c_str(),
                iteration_num, str_file_time);
    std::ofstream fp;
    fp.open(filename);
    if (fp.is_open()) {
        for (unsigned i = 0; i < data.size(); i++) {
            std::vector<int> values = data[i];
            for (unsigned j = 0; j < values.size(); j++) {
                fp << values[j];
                if (j != (values.size() - 1))
                    fp << ',';
            }
            fp << std::endl;
        }
        if (corrupted) {
#ifdef LOG
            char error_detail[200] = "";
            sprintf(error_detail,"#DUMP corrupted files dumped to %s", filename);
            log_error_detail(error_detail);
#endif
        }
    } else {
        printf("Could not open %s in save_corrupted_output()\n", filename);
    }
}

static std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector < std::string > elems;
    split(s, delim, elems);
    return elems;
}

static bool set_countains(std::vector<int> check, std::vector<std::vector<int> > src) {
    unsigned char cont = 0;
    for (size_t i = 0; i < src.size(); i++) {
        std::vector<int> temp = src[i];
        for (size_t j = 0; j < temp.size(); j++) {
            if (temp[j] == check[j])
                cont++;
        }
        if (cont == temp.size())
            return false;
        cont = 0;
    }
    return true;
}
#endif //CXX

double mysecond() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

#endif /* HELPFUL_H_ */
