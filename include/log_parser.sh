#!/bin/bash


if [ "$1" == "" ] || [ "$2" == "" ]; then
	    echo "usage '$0 <file_to_parser> <output.csv>'"
	    exit 1
fi

FILES=$1
OUT=$2

if [ -e $OUT ]; then
	echo "File $OUT exist. Do you really want to OVERWRITE the file?"
	select yn in "Yes" "No"; do
		case $yn in
			Yes ) break;;
			No ) exit 1;;
		esac
	done
fi

echo ""
echo " Output=$OUT"
echo " files to parser:$FILES"
echo ""

echo "LOG_FILE; SDC_ERROR_RATE; FI_ERROR_RATE; ITER_WITH_ERRORS; ACC_ERRORS; LAST_ACC_TIME; ABORT?; END?" > $OUT

for FILE in $FILES 
do
	echo "    parsing $FILE"
	ERRORS_COUNT=`grep \#SDC "$FILE" | wc -l`
	echo "    Iter with errors: $ERRORS_COUNT"
	LAST_ACC_KERNEL_TIME=`grep acc_time $FILE | tail -1 | awk '{print $4}' | awk -F : '{print $2}'`

	if [ $ERRORS_COUNT -ge 1 ]; then
		ACC_KERNEL_TIME=`grep \#SDC $FILE | tail -1 | awk '{print $4}' | awk -F : '{print $2}'`

		SDC_ERROR_RATE=`echo "$ERRORS_COUNT/$ACC_KERNEL_TIME" | bc -l | awk '{printf "%.8f", $0}'`
		FI_ERROR_RATE=`echo "1/$LAST_ACC_KERNEL_TIME" | bc -l | awk '{printf "%.8f", $0}'`

		ACC_ERR=`grep acc_err $FILE| tail -1 |awk '{print $6}' | awk -F : '{print $2}'`
	else
		ACC_KERNEL_TIME=0
		#LAST_ACC_KERNEL_TIME=`grep acc_time $FILE | tail -1 | awk '{print $4}' | awk -F : '{print $2}'`
		SDC_ERROR_RATE=0
		FI_ERROR_RATE=0
		ACC_ERR=0
	fi

	echo "    acc_time: $ACC_KERNEL_TIME"
	echo "    last_acc_time: $LAST_ACC_KERNEL_TIME"
	echo "    sdc_error_rate: $SDC_ERROR_RATE"
	echo "    fi_error_rate: $FI_ERROR_RATE"

	ABORT=`grep \#ABORT $FILE| wc -l`
	echo "    ABORT: $ABORT"
	END=`grep \#END $FILE| wc -l`
	echo "    END: $END"
	echo ""

	echo "$FILE; $SDC_ERROR_RATE; $FI_ERROR_RATE; $ERRORS_COUNT; $ACC_ERR; $LAST_ACC_KERNEL_TIME; $ABORT; $END" >> $OUT

done

echo " All done, it's been a pleasure. ciao"
echo ""

exit 0
