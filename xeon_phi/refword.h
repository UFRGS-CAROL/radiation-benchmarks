#include <inttypes.h>

#define REFWORD0            0
#define REFWORD1            0xFFFFFFFF
#define REFWORD2            1
#define REFWORD3            0x55555555

// =============================================================================
uint64_t string_to_uint64(char *string) {
    uint64_t result = 0;
    char c;

    for (  ; (c = *string ^ '0') <= 9 && c >= 0; ++string) {
        result = result * 10 + c;
    }
    return result;
}


//======================================================================
// Linear Feedback Shift Register using 32 bits and XNOR. Details at:
// http://www.xilinx.com/support/documentation/application_notes/xapp052.pdf
// http://www.ece.cmu.edu/~koopman/lfsr/index.html
uint32_t lfsr(){
    static uint64_t lfsr = 0x80000000;
    uint64_t bit;

    bit  = ~((lfsr >> 0) ^ (lfsr >> 10) ^ (lfsr >> 11) ^ (lfsr >> 30) ) & 1;
    lfsr =  (lfsr >> 1) | (bit << 31);

    return lfsr;
}

//======================================================================
inline void print_refword() {
    fprintf(stderr,"Refword option:\n");
    fprintf(stderr,"\t 0 = 0x%08x\n", REFWORD0);
    fprintf(stderr,"\t 1 = 0x%08x\n", REFWORD1);
    fprintf(stderr,"\t 2 = 0x%08x\n", REFWORD2);
    fprintf(stderr,"\t 3 = 0x%08x\n", REFWORD3);
    fprintf(stderr,"\t 4 = LFSR (RANDOM)\n");
};

//======================================================================
inline uint32_t get_refword(uint32_t opt) {
    switch(opt) {
        case 0: return REFWORD0;    break;
        case 1: return REFWORD1;    break;
        case 2: return REFWORD2;    break;
        case 3: return REFWORD3;    break;
        case 4: return lfsr();    break;
    }
};

