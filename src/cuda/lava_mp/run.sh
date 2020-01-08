set -e
set -x

make clean
make -j4 RELATIVEERROR=1

for ck in 1 194;
do
	for dmr in dmr dmrmixed none;
	do
		make test DMR=${dmr} OPERATIONNUM=${ck} > ${dmr}_${ck}_relative.csv
	done
done

make clean
make -j4 RELATIVEERROR=0

for ck in 1 194;
do
	for dmr in dmr dmrmixed none;
	do
		make test DMR=${dmr} OPERATIONNUM=${ck}> ${dmr}_${ck}_uint.csv
	done
done
