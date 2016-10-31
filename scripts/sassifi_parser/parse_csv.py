#!/usr/bin/python
import include
import os
import csv
import sys
import re

ERROR_MODEL_SIZE = len(include.EM_STR)
INSTRUCTION_SIZE = len(include.ASID_STR)

inst_type="inst"
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

def main(csv_input, logs_dir,cp):
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

    max_logs_count = 0
    #log_file,has_sdc,inj_kname,inj_kcount, inj_igid, inj_fault_model, inj_inst_id, inj_destination_id, inj_bit_location, finished
    for row in reader:
        if MAX_LOGS_SIZE == max_logs_count:
            break
        max_logs_count += 1
        #print row['log_file']
       # cp all good data to new folder
        if cp: os.system("cp "+ logs_dir + "/" + row['log_file'] +" good_logs/")
        it_inst_count = 8 #(int(row['inj_asid']) if inst_type == "inst" else int(row['inj_igid']))
        it_em_count = int(row['inj_fault_model'])
        #increase each instrction/error model count to have the final results
        if '1' in row['has_sdc']:
            sdc_inst_count[it_inst_count] += 1
            sdc_em_count[it_em_count] += 1
            sdcs += 1
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
           writer.writerow({'instruction': include.EM_STR[i], 'sdc_num': str(sdc_em_count[i]), 'total_inst_count': str(total_em_count[i]), 'crashes':str(crashes_em_count[i]), 'abort':str(abort_em_count[i])})

    writer.writerow({'instruction': '', 'sdc_num': ''})

    writer.writerow({'instruction': 'Total sdcs', 'sdc_num': str(sdcs)})
    writer.writerow({'instruction': 'Injected faults', 'sdc_num': str(total_faults)})
    print 'foi'
    runtime_average = sum(kern_time) /  len(kern_time)

    writer.writerow({'instruction': 'Average kernel runtime', 'sdc_num': str(runtime_average)})
    csvfile.close()
    print csv_input + " parsed"


def process_daniels_and_caios_log(csv_input, daniel_csv, is_daniel):
    print is_daniel
    csvfile = open(csv_input)
    reader = csv.DictReader(csvfile)
    daniel_input = open(daniel_csv)
    if is_daniel:
        reader_daniel = csv.DictReader(daniel_input, delimiter = ';',quoting=csv.QUOTE_NONE)
        #reader_daniel = csv.DictReader(daniel_input)
    else:
        reader_daniel = csv.DictReader(daniel_input)
    fieldnames = []
    for i in reader.fieldnames:
        fieldnames.append(i)

    for i in reader_daniel.fieldnames:
        fieldnames.append(i)
    my_lines = 0
    daniel_lines = 0

    output_csv = open("parsed_dc_"+csv_input, 'w')
    writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
    writer.writeheader()
    first = True
    daniel_rows = []
    logs_rows = []
    for row in reader_daniel:
        daniel_rows.append(row)
    for row in reader:
        logs_rows.append(row)

    #print writer.fieldnames
    count_logs = 0
    d = {}
    if is_daniel:
        for i in logs_rows:
            log_file = i['log_file']
            if '1' in i['has_sdc']:
                for j in daniel_rows:
                    logname = j['logFileName']
                    if logname in log_file and log_file not in d.values():
                        z = j.copy()
                        z.update(i)
                        d = z.copy()
                        writer.writerow(z)
                        count_logs += 1
                        #print d
                        break

    else:
        for i in logs_rows :
            log_file = i['log_file'].replace("./","")
            if '1' in i['has_sdc']:
                for j in daniel_rows:
                    logname = j['Logname']
                    if  log_file in logname:
                        z = j.copy()
                        z.update(i)
                        d = z.copy()
                        writer.writerow(z)
                        count_logs += 1
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
    print "For merge and parse Daniel's log <csv_input> <daniel_csv> <is_daniel> <caio | none>"

if __name__ == "__main__":
    parameter = sys.argv[1:]
    #()
    if len(parameter) < 3:
        usage()
    else:
        print parameter[3]
        if parameter[3] != 'caio':
            inst_type = (parameter[3] if parameter[3] == 'rf' else 'inst')
            main(parameter[0], parameter[1], (True if parameter[2] == 'cp' else False))
        #():
        else:
            process_daniels_and_caios_log(parameter[0], parameter[1], (True if parameter[2] == '1' else False))
