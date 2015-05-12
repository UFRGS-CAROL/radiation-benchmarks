#ifdef __cplusplus
extern "C" {
#endif


int isInteger(char *str) {


    if (*str == '\0') {
        return 0;
    }


    for(; *str != '\0'; str++) {
        if (*str < 48 || *str > 57) {	// digit characters (need to include . if checking for float)
            return 0;
        }
    }


    return 1;
}


#ifdef __cplusplus
}
#endif
