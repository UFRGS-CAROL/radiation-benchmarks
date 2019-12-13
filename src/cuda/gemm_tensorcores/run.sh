set -e
set -x

make clean
make -j4 BUILDRELATIVEERROR=1

for ck in 1 34;
do
	for dmr in full mixed none;
	do
		make test_gemm DMR=${dmr} CHECKBLOCK=${ck} > ${dmr}_${ck}_relative.csv
	done
done

make clean
make -j4 BUILDRELATIVEERROR=0


for ck in 1 34;
do
	for dmr in full mixed none;
	do
		make test_gemm DMR=${dmr} CHECKBLOCK=${ck}> ${dmr}_${ck}_uint.csv
	done
done
