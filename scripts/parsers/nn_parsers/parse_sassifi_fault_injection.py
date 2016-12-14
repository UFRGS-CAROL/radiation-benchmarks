#!/usr/bin/python
import include
import os
import csv
import sys
import re
from collections import Counter

ERROR_MODEL_SIZE = len(include.EM_STR)
INSTRUCTION_SIZE = len(include.ASID_STR)

inst_type="rf"
MAX_LOGS_SIZE=9999999999

def check_crash(log_filename):
    log_file = open(log_filename)
    regexp = re.compile(r'.*KerTime.*?([0-9.-]+)')
    there_is_end = 0
    there_is_abort = 0
    kernel_time = 0
    for line in log_file:
        match = regexp.match(line)
        if match:
            kernel_time = float(match.group(1))
        if 'END' in line:
            there_is_end = 1
        if 'ABORT' in line:
            there_is_abort = 1
    log_file.close()
    crash = 0
    if there_is_end == 0:
        crash = 1
    return (crash, there_is_abort, kernel_time)

def count_frequency(row_to_count):
    # row_to_count = [row[string_to_count] for row in reader]
    array_size = []
    for (k,v) in Counter(row_to_count).iteritems():
        array_size.append([k,v])

    return array_size

def parse_csv(csv_input, logs_dir, cp):
    global INSTRUCTION_SIZE, ERROR_MODEL_SIZE

    csvfile = open(csv_input)
    reader = csv.DictReader(csvfile)
    #count sdcs for each instruction
    sdc_inst_count = []
    #count sdcs for each error model
    sdc_em_count = []
    sdcs = 0
    #count total faults
    total_faults = 0
    #total faults per instruction
    total_inst_count = []
    #per error model
    total_em_count = []

    #count crashes and abort per instruction
    crashes_inst_count = []
    abort_inst_count = []
    #count crashes and abort per error model
    crashes_em_count = []
    abort_em_count = []

    #count for each injection type the kernel occurrence
    sdc_em_count_per_kernel = {}
    sdc_inst_count_per_kernel = {}
    inj_em_count_per_kernel = {}
    inj_inst_count_per_kernel = {}
    # kernel_array = count_frequency(reader,"inj_kname")

    #kernel time
    kern_time = []

    total_crashes = 0
    total_aborts = 0
    print "Parsing " + csv_input
    #separate the good data
    if cp: os.system("mkdir -p ./good_logs")
    for i in range(0,INSTRUCTION_SIZE):
        sdc_inst_count.append(0)
        total_inst_count.append(0)
        crashes_inst_count.append(0)
        abort_inst_count.append(0)

    for i in range(0,ERROR_MODEL_SIZE):
        sdc_em_count.append(0)
        total_em_count.append(0)
        crashes_em_count.append(0)
        abort_em_count.append(0)

    # for i in kernel_array:
    #     kernel = str(i[0])
    #     sdc_em_count_per_kernel[kernel] = sdc_em_count


    max_logs_count = 0
    #log_file,has_sdc,inj_kname,inj_kcount, inj_igid, inj_fault_model, inj_inst_id, inj_destination_id, inj_bit_location, finished
    for row in reader:
        if MAX_LOGS_SIZE == max_logs_count:
            break
        max_logs_count += 1
        #print row['log_file']
       # cp all good data to new folder
        if cp: os.system("cp "+ logs_dir + "/" + row['log_file'] +" good_logs/")
        it_inst_count = 8

        if 'inst' in inst_type:
            it_inst_count = int(row['inj_igid'])

        it_em_count = int(row['inj_fault_model'])
        #increase each instrction/error model count to have the final results
        if '1' in row['has_sdc']:
            sdc_inst_count[it_inst_count] += 1
            sdc_em_count[it_em_count] += 1
            sdcs += 1
            #count em per kernel
            if row["inj_kname"] not in sdc_em_count_per_kernel:
                sdc_em_count_per_kernel[row["inj_kname"]] = []
                for x in range(0, ERROR_MODEL_SIZE):
                    sdc_em_count_per_kernel[row["inj_kname"]].append(0)

            if row["inj_kname"] not in sdc_inst_count_per_kernel:
                sdc_inst_count_per_kernel[row["inj_kname"]] = []
                for x in range(0, INSTRUCTION_SIZE):
                    sdc_inst_count_per_kernel[row["inj_kname"]].append(0)

            sdc_em_count_per_kernel[row["inj_kname"]][it_em_count] += 1
            sdc_inst_count_per_kernel[row["inj_kname"]][it_inst_count] += 1

        if row["inj_kname"] not in inj_em_count_per_kernel:
            inj_em_count_per_kernel[row["inj_kname"]] = 0

        if row["inj_kname"] not in inj_inst_count_per_kernel:
            inj_inst_count_per_kernel[row["inj_kname"]] = 0

        inj_em_count_per_kernel[row["inj_kname"]] += 1
        inj_inst_count_per_kernel[row["inj_kname"]] += 1
        #check crash info for each file
        (crash, abort, kertime) = check_crash(logs_dir + "/" + row['log_file'])
        if crash > 1 or abort > 1:
            print 'Some error in the log files'
            exit(1)
        crashes_inst_count[it_inst_count] += crash
        abort_inst_count[it_inst_count] += abort
        crashes_em_count[it_em_count] += crash
        abort_em_count[it_em_count] += abort
        total_crashes += crash
        kern_time.append(kertime)
        total_faults += 1
        #print row['inj_asid'] + " " + row['inj_fault_model']

        total_inst_count[it_inst_count] += 1
        total_em_count[it_em_count] += 1

    csvfile.close();
    #---------------------------------------------------------------
    #print instruction histogram
    csvfile = open('parse_'+csv_input, 'w')
    fieldnames = ['instruction', 'sdc_num', 'total_inst_count', 'crashes', 'abort']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(0, INSTRUCTION_SIZE):
        writer.writerow({'instruction': include.ASID_STR[i], 'sdc_num': str(sdc_inst_count[i]), 'total_inst_count': str(total_inst_count[i]),
        'crashes': str(crashes_inst_count[i]), 'abort': str(abort_inst_count[i])})

    writer.writerow({'instruction': '', 'sdc_num': '', 'total_inst_count':''})
    writer.writerow({'instruction': 'error_model', 'sdc_num': 'sdc_num', 'total_inst_count':'total_em_count'})
    for i in range(0, ERROR_MODEL_SIZE):
		writer.writerow({'instruction': include.EM_STR[i], 'sdc_num': str(sdc_em_count[i]), 'total_inst_count': str(total_em_count[i]),
                            'crashes':str(crashes_em_count[i]), 'abort':str(abort_em_count[i])})

    writer.writerow({'instruction': '', 'sdc_num': ''})

    writer.writerow({'instruction': 'Total sdcs', 'sdc_num': str(sdcs)})
    writer.writerow({'instruction': 'Injected faults', 'sdc_num': str(total_faults)})
    # print kern_time
    runtime_average = sum(kern_time) /  len(kern_time)

    writer.writerow({'instruction': 'Average kernel runtime', 'sdc_num': str(runtime_average)})

    writer.writerow({'instruction': 'error model', 'sdc_num': 'SDCs', 'total_inst_count':''})

    for kernel in inj_em_count_per_kernel:
        # kernel = str(i[0])
        writer.writerow({'instruction': kernel})
        if kernel in sdc_em_count_per_kernel:
            err_list = sdc_em_count_per_kernel[kernel]
            for j in range(ERROR_MODEL_SIZE):
                writer.writerow({'instruction': include.EM_STR[j], 'sdc_num': str(err_list[j])})

    writer.writerow({'instruction': '', 'sdc_num': '', 'total_inst_count':''})
    writer.writerow({'instruction': 'Instructions', 'sdc_num': 'SDCs', 'total_inst_count':''})

    for kernel in inj_inst_count_per_kernel:
        writer.writerow({'instruction': kernel})
        if kernel in sdc_inst_count_per_kernel:
            err_list = sdc_inst_count_per_kernel[kernel]
            for j in range(INSTRUCTION_SIZE):
                writer.writerow({'instruction': include.ASID_STR[j], 'sdc_num': str(err_list[j])})

    writer.writerow({'instruction': '', 'total_inst_count':''})
    writer.writerow({'instruction': 'kernel', 'sdc_num': 'injected faults', 'total_inst_count':''})
    for kernel in inj_inst_count_per_kernel:
        writer.writerow({'instruction': kernel, 'sdc_num': inj_em_count_per_kernel[kernel]})
    csvfile.close()
    print csv_input + " parsed"

def process_daniels_and_caios_log(csv_input, daniel_csv, is_daniel):
    print is_daniel
    csvfile = open(csv_input)
    reader = csv.DictReader(csvfile)
    daniel_input = open(daniel_csv)
    # if is_daniel:
    #     reader_daniel = csv.DictReader(daniel_input, delimiter=';', quoting=csv.QUOTE_NONE)
    #     # reader_daniel = csv.DictReader(daniel_input)
    # else:
    reader_daniel = csv.DictReader(daniel_input)
    fieldnames = []
    for i in reader.fieldnames:
        fieldnames.append(i)

    for i in reader_daniel.fieldnames:
        fieldnames.append(i)
    my_lines = 0
    daniel_lines = 0

    output_csv = open("parsed_dc_" + csv_input, 'w')
    writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
    writer.writeheader()
    first = True
    daniel_rows = []
    logs_rows = []
    for row in reader_daniel:
        daniel_rows.append(row)
    for row in reader:
        logs_rows.append(row)

    # print writer.fieldnames
    count_logs = 0
    d = {}
    index_str_log = 'log_file'
    has_sdc = 'has_sdc'
    if is_daniel == 'd':
        log_file_name = 'logFileName'
    elif is_daniel == 'c':
        log_file_name = 'Logname'
    elif is_daniel == 'l':
        log_file_name = '_File_name'


    if is_daniel != 'none':
        for i in logs_rows:
            log_file = i[index_str_log]
            if '1' in i[has_sdc]:
                for j in daniel_rows:
                    logname = j[log_file_name]
                    if logname in log_file and log_file not in d.values():
                        z = j.copy()
                        z.update(i)
                        d = z.copy()
                        writer.writerow(z)
                        count_logs += 1
                        # print d
                        break

    print "Parsed " + str(count_logs)
    csvfile.close();
    daniel_input.close()
    output_csv.close()

# except:
   #     e = sys.exc_info()[0]
   #     #write_to_page( "<p>Error: %s</p>" % e )
   #     print e
def usage():
    print "For parse raw data <csv_input> <logs_dir> <cp | none> <caio | none>"
    print "For merge and parse Daniel's log <csv_input> <daniel_csv> <is_daniel> <c or d or l | none>"

if __name__ == "__main__":
    parameter = sys.argv[1:]
    #()
    if len(parameter) < 3:
        usage()
    else:
        print parameter[3]
        if parameter[3] != 'caio':
            inst_type = (parameter[3] if parameter[3] == 'rf' else 'inst')
            parse_csv(parameter[0], parameter[1], (True if parameter[2] == 'cp' else False))
        #():
        else:
            process_daniels_and_caios_log(parameter[0], parameter[1], parameter[2])
