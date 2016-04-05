import sys
import getopt
import pexpect
import random
import time
import configure_hog
import csv
import os
import subprocess

#hog config
N_CONTINUES_1x = int(100)
MAX_ARRAY_SIZE = 10
MAX_STEPS = 3

#------------------------
#CUDA-GDB commands
#------------------------
CUDA_GDB_PATH = "/usr/local/cuda/bin/cuda-gdb"
BREAKPOINT = "break "
STEPI = "stepi"
NEXT = "next"
PC = "print $pc"
RUN = "run"
CONTINUE = "continue"
CUDA_FUN_INFO = "cuda kernel block thread"
CUDA_THREAD_INFO = "info cuda threads block (2,0,0) thread (64,0,0)"
SWITCH = "cuda block (2,0,0) thread (64,0,0)"
YES = "y"
DELETE_BREAKPOINT = "delete breakpoint 1"
DELETE_ALL_BREAKS = "delete breakpoints"
DISABLE_ALL_BREAKS = 'disable breakpoints'
QUIT= "quit"
KILL = "kill"
ENTER = ""
EXIT = "Program exited normally"
CUDA_EXCEPTION = "CUDA_EXCEPTION_"
SIGTRAP = "Program received signal SIGTRAP"
SIGKILL = "SIGKILL"
FINISH = 'finish'
DISABLE_BREAK = 'disable breakpoint '
ENABLE_BREAK  = 'enable breakpoint '
#------------------------
#Expect collection
#------------------------

CUDA_GDB_EXPECT = "\(cuda-gdb\)"
CUDA_SYN_EXPECT = "__syncthreads\(\)"
CUDA_SYN_EXPECT_2 = " __syncthreads\(\)"
PC_EXPECT = "="
CUDA_FUN_INFO_EXPECT = " "
THREAD_CONTINUE_EXPECT = "---Type \<return\> to continue, or q \<return\> to quit---"
THREAD_CONTINUE_EXPECT_WERIED = "---Type \<return\> to continue, or q \<return\> to quit---stepi"
PROGRAM_KILLED = "Program terminated with signal SIGKILL, Killed"
NO_FOCUS = "Focus not set on any active CUDA kernel"
ALLBREAKPOINTS_RES = "Delete all breakpoints? \(y or n\)"
SET_REGISTER = "set $"


class ReturnObj(object):
    exception = ''
    def __init__(self, position, var):
        self.faults = {}
        self.faults["var"] = var
        if '*' in var:
            self.faults["var_is_array"] = "yes"
            self.faults["position"] = position
        else:
            self.faults["var_is_array"] = "no"
            self.faults["position"] = -1
        self.exeception = ""
        
    def set_exception(self, exception):
        self.exception = exception
            
        
def fault_injection(cuda_gdb_p, var):
    tem = '= 0'
    global SET_REGISTER, MAX_ARRAY_SIZE, CUDA_GDB_EXPECT
    
    #select registers to insert fault
    #llist = random.sample(configure_hog.register_list[0:N_REGISTERS], n_faults)
   

    #this is an array
    result = ReturnObj(random.randrange(1, MAX_ARRAY_SIZE), var)

    #for i in range(0, n_faults):
    #    string_to_send = SET_REGISTER + llist[i] + " = " + str(random.randrange(1, 1000000))      
    #    cuda_gdb_p.sendline(string_to_send)

    if '*' in var:
         string_to_send = "set variable "+var + " + " + str(result.faults["position"]) +") = "+ str(random.randrange(1, 1000000))
    else:
        string_to_send = "set variable "+var + " = "+ str(random.randrange(1, 1000000))
    print string_to_send
    cuda_gdb_p.sendline(string_to_send)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
            
    return result    

def count_continues(path, position, argument, breakpoint_location, choice, cuda_gdb_p):
    global CUDA_GDB_PATH, BREAKPOINT,KILL,QUIT,DELETE_BREAKPOINT
    global CUDA_FUN_INFO,PC,RUN,CONTINUE,CUDA_THREAD_INFO,ENTER
    global NEXT, CUDA_SYN_EXPECT_2
    global CUDA_GDB_EXPECT,PC_EXPECT,CUDA_FUN_INFO_EXPECT,THREAD_CONTINUE_EXPECT
    global CUDA_SYN_EXPECT,EXIT,NO_FOCUS,THREAD_CONTINUE_EXPECT_WERIED
    global SWITCH, PROGRAM_KILLED, YES, DELETE_ALL_BREAKS, ALLBREAKPOINTS_RES
    global MAX_FAULTS, MAX_STEPS, FINISH, DISABLE_ALL_BREAKS
    
    
    #cuda_gdb_p.sendline(ENABLE_BREAK + breakpoint_location)
    #---------------
    # run the program
    #---------------
    wc = cuda_gdb_p.sendline(RUN+argument)

    resend = cuda_gdb_p.expect([CUDA_GDB_EXPECT,THREAD_CONTINUE_EXPECT])
    #print cuda_gdb_p.read()
    if resend == 1:
        cuda_gdb_p.sendline()
    target = ''
    iterator = 0
    
    cuda_gdb_p.sendline(SWITCH)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    while 'is not being run' not in target:
        target = cuda_gdb_p.before
        #print target
        cuda_gdb_p.sendline(CONTINUE)   
        expected =  cuda_gdb_p.expect([CUDA_GDB_EXPECT,CUDA_SYN_EXPECT,THREAD_CONTINUE_EXPECT,PROGRAM_KILLED, 
                      CUDA_EXCEPTION, SIGTRAP, THREAD_CONTINUE_EXPECT_WERIED,CUDA_SYN_EXPECT_2,pexpect.TIMEOUT],timeout=60)
                      
        if expected != 0:
           cuda_gdb_p.sendline()
           
                
        ################################################################
        #fault injection pseudo technique
        if iterator == position:
            result = fault_injection(cuda_gdb_p, choice)            

            if THREAD_CONTINUE_EXPECT_WERIED in cuda_gdb_p.before:
                cuda_gdb_p.sendline(YES)
            
            cuda_gdb_p.sendline(DELETE_BREAKPOINT)
            cuda_gdb_p.expect(CUDA_GDB_EXPECT)
        iterator += 1

    return result
           
def main(argv):
    #os.system('rm -rf /var/radiation-benchmarks/log/*')
    os.system('rm -rf last_log.log')
    global N_CONTINUES_1x, BREAKPOINT, ENABLE_BREAK
    global SIGTRAP, CUDA_EXCEPTION, SIGKILL
    
    binary = configure_hog.binary_path
    try:
        opts, args = getopt.getopt(argv, "t:o:", ["times", "csv_output"])
    except getopt.GetoptError:
        print "Enter with -t <times> -o <csv>"
        sys.exit(2)
    
    for opt, arg in opts:
      if opt in ("-t", "--times"):
          times = int(arg)
      elif opt in ("-o", "--csv_output"):
          output = arg
    #register output
    csvfile = open(output, "wb") 
    spamwriter = csv.writer(csvfile, delimiter=',')
    #csv header
    spamwriter.writerow(['position', 'kernel', 'kernel_line', 'var', 'vet_position', 'var_is_array', 'SDC', 'finish', 'log_file'])

    final_writer = []
    parameter = configure_hog.parameter

    ####################################################################
    global CUDA_GDB_PATH, CUDA_GDB_EXPECT, THREAD_CONTINUE_EXPECT, QUIT
    #cuda_gdb_p = pexpect.spawn(CUDA_GDB_PATH+" "+binary)
    #cuda_gdb_p.logfile = sys.stdout
    #cuda_gdb_p.maxread = 1000000
    #cuda_gdb_p.setecho(False)
    #cuda_gdb_p.expect(CUDA_GDB_EXPECT)  
    print "Everything started ok"
    #################################################################### 

    #for kernel in configure_hog.kernel_names:
    

    for i in range(0, times):
            #path, position, argument, kernel, kernel_line
            for kernel in configure_hog.kernel_names:
                ####################################################################
                cuda_gdb_p = pexpect.spawn(CUDA_GDB_PATH+" "+binary)
                os.system("rm -rf last_log.log")
                fout = file('last_log.log','w')
                cuda_gdb_p.logfile = fout
                cuda_gdb_p.maxread = 1000000
                cuda_gdb_p.setecho(False)
                cuda_gdb_p.expect(CUDA_GDB_EXPECT)  
                #################################################################### 
                cuda_gdb_p.sendline(BREAKPOINT+ " "+ configure_hog.filename + ":"+str(configure_hog.kernel_start_line[kernel][0]))
                cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                #--------------------------------------------------------
                position = random.randrange(1, N_CONTINUES_1x)
                n_steps = random.randint(-1, MAX_STEPS)
                if 'resize_for_hog_kernel' in kernel:
                    kernel_line = configure_hog.kernel_start_line[kernel][0]
                else:
                    kernel_line = configure_hog.kernel_start_line[kernel][0] + n_steps
                                       
                choice = random.choice(configure_hog.critical_vars[kernel])
                
                print "Iteration "+ str(i) + " position " + str(position) + " kernel "+kernel + " var " + choice
                
                ret = count_continues(binary, position, parameter, str(configure_hog.kernel_start_line[kernel][1]), choice, cuda_gdb_p)
                
                time.sleep(1)
                #writes everything in the file
                proc = subprocess.Popen(" ls -dt /var/radiation-benchmarks/log/*.log | head -1", stdout=subprocess.PIPE, shell=True)
                (out, err) = proc.communicate()             
                there_is_sdc = 0
                there_is_end = 1
                out = out.strip()
                
                
                if 'No such file or' not in out:                
                    string_file = open(out).read()
                    fout_string = open('last_log.log','r').read()
                    print string_file
                                       
                    if  CUDA_EXCEPTION in fout_string or SIGTRAP in fout_string or SIGKILL in fout_string:
                        print "passou"
                        there_is_end = 0
                    
                    if 'SDC' in string_file:
                        there_is_sdc = 1
                    if 'ERR' in string_file:
                        there_is_sdc = 1
                    if 'END' not in string_file:
                        there_is_end = 0

                list_final = [position, kernel, kernel_line, choice, ret.faults["position"], ret.faults["var_is_array"], there_is_sdc, there_is_end, out]
                spamwriter.writerow(list_final)
                ########################################################
                cuda_gdb_p.sendline(QUIT)
                if cuda_gdb_p.isalive():
                    #time.sleep(20)
                    while cuda_gdb_p.terminate(force=False) != True:
                        print "trying terminate"
                        #time.sleep(1)
                
                ########################################################




    print "Everything finished ok"     
    csvfile.close()
 

if __name__ == "__main__" :
    main(sys.argv[1:])
