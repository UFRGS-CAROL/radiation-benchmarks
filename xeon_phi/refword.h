#include <inttypes.h>

#define REFWORD0            0
#define REFWORD1            1
#define REFWORD2            0x55555555
#define REFWORD3            0xFFFFFFFF

//=====================================================================
inline void print_refword() {
    printf("Refword option:\n");
    printf("\t 0 = 0x%08x\n", REFWORD0);
    printf("\t 1 = 0x%08x\n", REFWORD1);
    printf("\t 2 = 0x%08x\n", REFWORD2);
    printf("\t 3 = 0x%08x\n", REFWORD3);
};

//=====================================================================
inline uint32_t get_refword(uint32_t opt) {
    switch(opt) {
        case 0: return REFWORD0;    break;
        case 1: return REFWORD1;    break;
        case 2: return REFWORD2;    break;
        case 3: return REFWORD3;    break;
    }
};
