#include<stdio.h>
#include<stdlib.h>

int main() {
    int n = 16385;// + 1;
    //std::cout << "open array...\n";

    FILE *f_gold;
    f_gold = fopen("../../bin/gold_16384_th_228_pen_10", "rb");

    if ((f_gold == NULL)) {
        printf("Error opening files\n");
        exit(-3);
    }

    int * gold = (int *)malloc( n * n * sizeof(int) );
    printf("reading file\n");
    fread(gold, sizeof(int) * n * n, 1, f_gold);
    fclose(f_gold);
    int i, zero = 0;
    for(i=0; i< n; i++) {
        if(gold[i] == 0)
            zero++;
    }
    printf("# of zeros: %d\n",zero);
    free(gold);
}
