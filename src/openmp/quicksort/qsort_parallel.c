#include<stdio.h>
#include<omp.h>
#include<stdio.h>
#include<time.h>

int a[10000009];
void qsort_serial(int l,int r){
	if(r>l){
		int pivot=a[r],tmp;
		int less=l-1,more;
		for(more=l;more<=r;more++){
			if(a[more]<=pivot){
				less++;
				 tmp=a[less];
				a[less]=a[more];
				a[more]=tmp;
			}
		}
		qsort_serial(l,less-1);
		qsort_serial(less+1,r);
	}
}
void qsort_parallel(int l,int r){
	if(r>l){

		int pivot=a[r],tmp;
		int less=l-1,more;
		for(more=l;more<=r;more++){
			if(a[more]<=pivot){
				less++;
				tmp=a[less];
				a[less]=a[more];
				a[more]=tmp;
			}
		}
		if((r-l)<1000){
			qsort_serial(l,less-1);
			qsort_serial(less+1,r);
		}
		else{
			#pragma omp task
			qsort_parallel(l,less-1);
			#pragma omp task
			qsort_parallel(less+1,r);
			#pragma omp taskwait
		}
	}
}
int main(){
	int n,i;

	n=10000000; //increased the value of n
	int range=100000;
	srand(time(NULL));
	for( i=0;i<n;i++)
		a[i]=rand()%range;

	#pragma omp parallel	
	#pragma omp single
	qsort_parallel(0,n-1);

	
	return 0;
}
