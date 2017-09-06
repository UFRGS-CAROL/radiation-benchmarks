#include "support/common.h"
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Params ---------------------------------------------------------------------
struct Params {

    const char *file_name;
    int         in_size_i;
    int         in_size_j;
    int         out_size_i;
    int         out_size_j;

    Params(int argc, char **argv) {

        file_name     = "input/control.txt";
        in_size_i = in_size_j = 3;
        out_size_i = out_size_j = 300;
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:f:m:n:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'f': file_name     = optarg; break;
            case 'm': in_size_i = in_size_j = atoi(optarg); break;
            case 'n': out_size_i = out_size_j = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }

    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./bs [options]"
                "\nBenchmark-specific options:"
                "\n    -f <F>    name of input file with control points (default=input/control.txt)"
                "\n    -m <N>    input size in both dimensions (default=3)"
                "\n    -n <R>    output resolution in both dimensions (default=300)"
                "\n");
    }
};

void read_input(XYZ *in, const Params &p) {

    // Open input file
    FILE *f = NULL;
    f       = fopen(p.file_name, "r");
    if(f == NULL) {
        puts("Error opening file");
        exit(-1);
    }

    // Store points from input file to array
    int k = 0, ic = 0;
    XYZ v[10000];
#if DOUBLE_PRECISION
    while(fscanf(f, "%lf,%lf,%lf", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
#else
    while(fscanf(f, "%f,%f,%f", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
#endif
    {
        ic++;
    }
    for(int i = 0; i <= p.in_size_i; i++) {
        for(int j = 0; j <= p.in_size_j; j++) {
            in[i * (p.in_size_j + 1) + j].x = v[k].x;
            in[i * (p.in_size_j + 1) + j].y = v[k].y;
            in[i * (p.in_size_j + 1) + j].z = v[k].z;
            //k++;
            k = (k + 1) % 16;
        }
    }
}
// BezierBlend (http://paulbourke.net/geometry/bezier/)
inline T BezierBlend(int k, T mu, int n) {
    int nn, kn, nkn;
    T   blend = 1;
    nn        = n;
    kn        = k;
    nkn       = n - k;
    while(nn >= 1) {
        blend *= nn;
        nn--;
        if(kn > 1) {
            blend /= (T)kn;
            kn--;
        }
        if(nkn > 1) {
            blend /= (T)nkn;
            nkn--;
        }
    }
    if(k > 0)
        blend *= pow(mu, (T)k);
    if(n - k > 0)
        blend *= pow(1 - mu, (T)(n - k));
    return (blend);
}

// Sequential implementation for comparison purposes
inline void BezierCPU(XYZ *inp, XYZ *outp, int NI, int NJ, int RESOLUTIONI, int RESOLUTIONJ) {
    int i, j, ki, kj;
    T   mui, muj, bi, bj;
    for(i = 0; i < RESOLUTIONI; i++) {
        mui = i / (T)(RESOLUTIONI - 1);
        for(j = 0; j < RESOLUTIONJ; j++) {
            muj     = j / (T)(RESOLUTIONJ - 1);
            XYZ out = {0, 0, 0};
            for(ki = 0; ki <= NI; ki++) {
                bi = BezierBlend(ki, mui, NI);
                for(kj = 0; kj <= NJ; kj++) {
                    bj = BezierBlend(kj, muj, NJ);
                    out.x += (inp[ki * (NJ + 1) + kj].x * bi * bj);
                    out.y += (inp[ki * (NJ + 1) + kj].y * bi * bj);
                    out.z += (inp[ki * (NJ + 1) + kj].z * bi * bj);
                }
            }
            outp[i * RESOLUTIONJ + j] = out;
        }
    }
}


int main(int argc, char **argv) {

    const Params p(argc, argv);
    int in_size   = (p.in_size_i + 1) * (p.in_size_j + 1) * sizeof(XYZ);
    int out_size  = p.out_size_i * p.out_size_j * sizeof(XYZ);

    XYZ *h_in = (XYZ *)malloc(in_size);
    XYZ *gold = (XYZ *)malloc(out_size);

    read_input(h_in, p);

    BezierCPU(h_in, gold, p.in_size_i, p.in_size_j, p.out_size_i, p.out_size_j);

	FILE *finput;
    char filename[100];
    snprintf(filename, 100, "gold_%d",p.out_size_i); // Gold com a resolução 
    if (finput = fopen(filename, "wb")) {
        fwrite(gold, out_size, 1 , finput);
    } else {
        printf("Error writing gold file");
        exit(1);
    }
    fclose(finput);

}

