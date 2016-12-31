//teste gold pointer

#include "log_processing.h"

void printGold(ProbArray gold, long plist_iteration){
	long i,j;
	for (i = 0; i < gold.total_size; ++i) {
		box tmp_pb = gold.boxes[i];
		char error_detailb[300];
		sprintf(error_detailb, "image_list_position: [%ld] boxes: [%d] "
					" x_r: %f"
					" y_r: %f"
					" w_r: %f"
					" h_r: %f\n", plist_iteration,
					i, tmp_pb.x, tmp_pb.y,tmp_pb.w, tmp_pb.h);
		for (j = 0; j < gold.classes; ++j) {
			char error_detail[300];
			
			sprintf(error_detail, "image_list_position: [%ld] probs: [%d,%d] "
									" prob_r: %f\n", plist_iteration,
							i, j, gold.probs[i][j]);
		     if (gold.probs[i][j] >= 0.2)
				printf("%s %s", error_detail, error_detailb);
				
		}
	}
}


int main(){
	int classes = 20;
	int tsize = 147;
	int plist = 1000;
	
	GoldPointers gold_ptr = new_gold_pointers(classes, tsize, plist,
	"/home/familia/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40/darknet"
	"/gold.caltech.critical.1K.test", 
	"rb");
	read_yolo_gold(&gold_ptr);
	printf("passou do read\n");
	printGold(*gold_ptr.pb_gold, 574);
	
	free_gold_pointers(&gold_ptr);
	printf ("passou\n");
	return 0;
}
