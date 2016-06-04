#!/usr/bin/python

import sys
import getopt
#import pexpect
import random
import time
#import configure_hog
import csv
import os
import subprocess
import support_classes as sp
import configure_hog

#hog config
N_CONTINUES_1x = int(5)
MAX_ARRAY_SIZE = 10
MAX_STEPS = 3


#cuda signals
#termination, program, alarm, asynchronous, job, operation error, miscellaneous, si
SIGNALS = ['SIGKILL','SIGTERM','SIGINT', 'SIGQUIT','SIGHUP',                                   #termination codes
'SIGFPE', 'SIGILL','SIGSEGV', 'SIGBUS', 'SIGABRT', 'SIGIOT', 'SIGTRAP','SIGEMT','SIGSYS',     #program codes
'SIGALRM', 'SIGVTALRM', 'SIGPROF',                                                            #alarm codes
'SIGIO', 'SIGURG', 'SIGPOLL',                                                                 #asynchronous codes
'SIGCHLD', 'SIGCLD', 'SIGCONT', 'SIGSTOP', 'SIGTSTP', 'SIGTTIN', 'SIGTTOU',                    #job control 
'SIGPIPE', 'SIGLOST', 'SIGXCPU', 'SIGXFSZ',                                                     #operation codes
'SIGUSR1', 'SIGUSR2', 'SIGWINCH', 'SIGINFO',                                                    #miscellaneous codes
'strsignal', 'psignal',                                                                         #signal messages
#cuda signals
'CUDA_EXCEPTION_0','CUDA_EXCEPTION_1','CUDA_EXCEPTION_2','CUDA_EXCEPTION_3','CUDA_EXCEPTION_4','CUDA_EXCEPTION_5',
'CUDA_EXCEPTION_6','CUDA_EXCEPTION_7','CUDA_EXCEPTION_8','CUDA_EXCEPTION_9','CUDA_EXCEPTION_10','CUDA_EXCEPTION_11',
'CUDA_EXCEPTION_12','CUDA_EXCEPTION_13','CUDA_EXCEPTION_14','CUDA_EXCEPTION_15']

def fault_injection(var, printed, array_i): #cuda_gdb_p, var):
    #this is an array
    result = sp.ReturnObj(array_i, var, 0)
    max_less_equal_1 = 5.5
    min_less_equal_1 = 1.5
    max_less_equal_5 = 2.5
    mix_less_equal_5 = 0.1
    max_all = 0.8
    min_all = 0.1
    string_to_send =''
    if '(' in var:
        var_string = var + " + "+ str(result.faults["position"]) +")"
        l = []
        for t in printed.split():
            try:
                l.append(float(t))
            except ValueError:
                pass

        if abs(l[0]) <= 1:
            num = random.uniform(min_less_equal_1, max_less_equal_1)
            result.faults["set_val"] = str(int(num)) if  l[0].is_integer() else str(num)
        elif abs(l[0]) <= 10:
            num = random.uniform(mix_less_equal_5, max_less_equal_5)
            result.faults["set_val"] = str(int(num  * l[0])) if  l[0].is_integer() else str(num * l[0] )
        else:
            num = random.uniform(min_all, max_all)
            result.faults["set_val"] = str(int(num  * l[0])) if  l[0].is_integer() else str(num * l[0] )

        if '{' in printed:
            string_to_send = "set variable "+ var_string+ ".y " + " = " + result.faults["set_val"]
        else:
            string_to_send = "set variable "+ var_string + " = " + result.faults["set_val"]
        result.faults["old_value"] = l[0]
    else:
        l = []
        for t in printed.split():
            try:
                l.append(float(t))
            except ValueError:
                pass
        if abs(l[0]) <= 1:
            num = random.uniform(min_less_equal_1, max_less_equal_1)
            result.faults["set_val"] = str(int(num)) if  l[0].is_integer() else str(num)
            string_to_send = "set variable "+var + " = " + result.faults["set_val"] 
        elif abs(l[0]) <= 10:
            num = random.uniform(mix_less_equal_5, max_less_equal_5)
            result.faults["set_val"] = str(int(num  * l[0])) if  l[0].is_integer() else str(num * l[0] )
            string_to_send = "set variable "+var + " = " + var + " * " + result.faults["set_val"] 
        else:
            num = random.uniform(min_all, max_all)
            result.faults["set_val"] = str(int(num  * l[0])) if  l[0].is_integer() else str(num * l[0] )
            string_to_send = "set variable "+var + " = " + var + " * " + result.faults["set_val"] 
     
        
        result.faults["old_value"] = l[0]
    result.set_variable(string_to_send)
    return result    


def last_step(output_csv, position, ret, kernel, kernel_line, choice, log_name):
        #register output
        global SIGNALS
        csvfile = open(output_csv, "a") 
        spamwriter = csv.writer(csvfile, delimiter=',')
        #writes everything in the file
        proc = subprocess.Popen("ls -dt /var/radiation-benchmarks/log/*.log | head -1", stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()             
        there_is_sdc = 0
        there_is_end = 1
        
        out = out.strip()
        if 'No such file or' not in out:                
            string_file = open(out).read()
            fout_string = open(configure_hog.path + log_name,'r').read()
                               
            if 'SDC' in string_file:
                there_is_sdc = 1
            if 'ERR' in string_file:
                there_is_sdc = 1
            
            sig_kill = []
            for message in SIGNALS:
                mess_temp = message + ','
                if mess_temp in fout_string:
                    mess_temp.replace(',', '#')
                    sig_kill.append(mess_temp)
                    there_is_end = 0
                if message in fout_string:
                    sig_kill.append(message)
                    there_is_end = 0
                    
        sig_kill_str = str(sig_kill).replace(',', ' ')
        print "Is there SDC "+ str(there_is_sdc)
        print "It finished " + str(there_is_end)
#    spamwriter.writerow(['position', 'kernel', 'kernel_line',  'var', 'new_value', 'old_value', 'vet_position', 'var_is_array', 'SDC', 'finish', 'sig_kill', 'log_file'])
        list_final = [position, kernel, kernel_line, choice, ret.faults["set_val"], ret.faults["old_value"], ret.faults["position"], ret.faults["var_is_array"], there_is_sdc, there_is_end,sig_kill, out]
        spamwriter.writerow(list_final)
        
        csvfile.close()
        os.system("mv "+ out + " /var/radiation-benchmarks/log/" + kernel)
           
def main(argv):
    binary = configure_hog.binary_path
    times = 1
    output = configure_hog.path + "test.csv"
    
    try:
        opts, args = getopt.getopt(argv, "t:o:", ["times", "csv_output"])
    except getopt.GetoptError:
        print "Chosing default parameters 1 time and test.csv, for other parameters use -t <times> -o <output>"

        
    for opt, arg in opts:
      if opt in ("-t", "--times"):
          times = int(arg)
      elif opt in ("-o", "--csv_output"):
          output = arg
    
    #csv header
    #spamwriter.writerow(['position', 'kernel', 'kernel_line', 'var', 'new_value', 'old_value', 'vet_position', 'var_is_array', 'SDC', 'finish', 'sig_kill', 'log_file'])

    final_writer = []
    parameter = configure_hog.parameter

    ####################################################################
    print "The process has started ok"
    #################################################################### 

    try:
        for i in range(0, times):
            for kernel in configure_hog.kernel_names:
                #Profile the kernel
                #init the cuda gdb process
                position = random.randrange(1, N_CONTINUES_1x)
                n_steps = random.randint(0, configure_hog.kernel_start_line[kernel][1])
                kernel_line = configure_hog.kernel_start_line[kernel][0] + n_steps
                array_i = random.randrange(1, MAX_ARRAY_SIZE)
                choice = random.choice(configure_hog.critical_vars[kernel])
                ########################################################
                #print var in a fault free gdb execution
                if '(' in choice:
                    temp_var = choice + '+' + str(array_i) +  ')'
                else:
                    temp_var = choice
                faultFreeObj = sp.SupportFile(configure_hog.path, "hog_without_log", configure_hog.filename + ":" + str(kernel_line), configure_hog.parameter, position, "", "print " + temp_var)
                faultFreeObj.generate_gdb_file(configure_hog.path, "gdb_fault_free.gdb")
                faultFreeObj.run("fault_free.log")
                printed = faultFreeObj.get_printed_string()               

                #select the values changed
                ret = fault_injection(choice, printed, array_i)
                print "Iteration "+ str(i) + " position " + str(position) + " kernel "+kernel + " var " + choice + " value " + str(ret.faults["set_val"]) + " old value "+ str(ret.faults["old_value"])
                ################################################################
                #perform the fault injection without
                faultObj = sp.SupportFile(configure_hog.path, "hog_without", configure_hog.filename + ":" + str(kernel_line), configure_hog.parameter, position, ret.faults["set_variable_string"], "")
                faultObj.generate_gdb_file(configure_hog.path, "gdb_fault_injection.gdb")
                faultObj.run("fault_injection.log")
                ###############################################################
                #check the output file
                last_step(output,position, ret, kernel, kernel_line,choice, "fault_injection.log")
                
        print "Everything finished ok"
    except Exception, e:
        print "some error happened"
        print sys.exc_info()[0]
        raise
 

if __name__ == "__main__" :
    main(sys.argv[1:])

