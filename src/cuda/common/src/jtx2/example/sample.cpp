
#include <stdio.h>
#include "../../../include/JTX2Inst.h"
#include <unistd.h>

int main(){
	rad::JTX2Inst measure(0);

	measure.start_collecting_data();

	printf("SLEEPING FOR 5 seconds\n");
	sleep(5);
	measure.end_collecting_data();




	return 0;
}
