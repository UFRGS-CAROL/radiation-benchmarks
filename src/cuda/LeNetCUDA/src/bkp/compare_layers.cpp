/*
 * compare_layers.cpp
 *
 *  Created on: Jun 27, 2017
 *      Author: carol
 */

#include <vector>
#include <iostream>
#include <cstdio>

using namespace std;

int size_file(FILE *fp){
	fseek(fp, 0L, SEEK_END);
	auto sz = ftell(fp);
	rewind(fp);
	return sz;
}


bool compare(FILE *fgpu, FILE *fcpu){

	while(!feof(fgpu) || !feof(fcpu)){
		unsigned char i_gpu, i_cpu;
		fread(&i_gpu, sizeof(unsigned char), 1, fgpu);
		fread(&i_cpu, sizeof(unsigned char), 1, fcpu);

		float diff = float(i_gpu) / float(i_cpu);
		cout << i_gpu << " " << i_cpu  << "\n";
		if(diff > 1e-3){
			cout << "Pau, layer diff " << diff << "\n";
			return false;
		}
	}
	return true;
}

int main(){

	for(auto t = 0; t < 6; t++){
		FILE *fgpu;
		FILE *fcpu;

		char temp[200];

		sprintf(temp, "layer_%d_cpu_layer.lay", t);
		fcpu = fopen(temp, "rb");

		cout << temp << "\n";

		sprintf(temp, "layer_%d_gpu_layer.lay", t);
		fgpu = fopen(temp, "rb");
		cout << temp << "\n";

		if(fcpu == NULL || fgpu == NULL){
			cout << "Pau nos arquivos\n";
			exit(-1);
		}

		auto cpu_siz = size_file(fcpu);
		auto gpu_siz = size_file(fgpu);


		cout << cpu_siz << " " << gpu_siz << "\n";

		if (cpu_siz != gpu_siz){
			cout << "Pau, layers not equal\n";
			fclose(fgpu);
			fclose(fcpu);
			break;
		}

		if (!compare(fgpu, fcpu)){
			fclose(fgpu);
			fclose(fcpu);
			break;
		}

		fclose(fgpu);
		fclose(fcpu);

	}

return 0;
}
