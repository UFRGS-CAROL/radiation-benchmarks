import sys
import getopt
import pexpect
import random
import time
import configure_hog
import csv

#hog config
N_CONTINUES_9x = int(32218 / 4)
N_CONTINUES_4x = int(29026 / 4)
N_CONTINUES_1x = int(13108 / 4)
N_REGISTERS = 50
MAX_FAULTS  = 5
MAX_STEPS   = 5

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

DELETE_BREAKPOINT = "delete breakpoint 1"
QUIT= "quit"
KILL = "kill"
ENTER = ""
EXIT = "Program exited normally"
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
NO_FOCUS = "Focus not set on any active CUDA kernel"
SET_REGISTER = "set $"


class ReturnObj(object):
    def __init__(self, reg_list, n_faults, n_steps):
        self.reg_list = reg_list
        self.n_faults = n_faults
        self.n_steps  = n_steps

def fault_injection(n_steps, n_faults, cuda_gdb_p):
    global SET_REGISTER, N_REGISTERS
    #go inside the kernel till steps finish
    for i in range(0 , n_steps):
        cuda_gdb_p.sendline(NEXT)
    
    #select registers to insert fault
    llist = random.sample(configure_hog.register_list[0:N_REGISTERS], n_faults)
    result = ReturnObj(llist, n_faults, n_steps)
    for i in range(0, n_faults):
        string_to_send = SET_REGISTER + llist[i] + " = " + str(random.randrange(1, 100000000))
        #print string_to_send
        cuda_gdb_p.sendline(string_to_send)
    return result    

def count_continues(path, position, argument, breakpoint_location):
    global CUDA_GDB_PATH, BREAKPOINT,KILL,QUIT,DELETE_BREAKPOINT
    global CUDA_FUN_INFO,PC,RUN,CONTINUE,CUDA_THREAD_INFO,ENTER
    global  NEXT, CUDA_SYN_EXPECT_2
    global CUDA_GDB_EXPECT,PC_EXPECT,CUDA_FUN_INFO_EXPECT,THREAD_CONTINUE_EXPECT,CUDA_SYN_EXPECT,EXIT,NO_FOCUS,THREAD_CONTINUE_EXPECT_WERIED
    global SWITCH
    global MAX_FAULTS, MAX_STEPS
    cuda_gdb_p = pexpect.spawn(CUDA_GDB_PATH+" "+path)
    cuda_gdb_p.maxread = 1000000
    cuda_gdb_p.setecho(False)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)  
    #---------------
    # set breakpoint
    #---------------
    cuda_gdb_p.sendline(BREAKPOINT+" "+breakpoint_location)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #---------------
    # run the program
    #---------------
    wc = cuda_gdb_p.sendline(RUN+argument)
    resend = cuda_gdb_p.expect([CUDA_GDB_EXPECT,THREAD_CONTINUE_EXPECT])
    if resend == 1:
       cuda_gdb_p.sendline()
    rawstr = cuda_gdb_p.before

    lines = rawstr.split("\r\n")
   
    #------------------------------
    # check the current PC
    #------------------------------
    cuda_gdb_p.sendline(CUDA_FUN_INFO)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    #logger.info("KERNEL INFO: "+cuda_gdb_p.before)
    cuda_gdb_p.sendline(PC)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    value = cuda_gdb_p.before.lstrip().rstrip("\r\n").split(PC_EXPECT)
    #logger.info("PC is "+value[len(value)-1])
    target = ""
    temp = ""
    cuda_gdb_p.sendline(SWITCH)
    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
    iterator = 0
    while "is not being run" not in target or "is not being run" not in temp:
        iterator += 1
        j = -1
        flag_step = 0
        flag_info = 0
        cuda_gdb_p.sendline(CONTINUE)
        while flag_step == 0:
            j = cuda_gdb_p.expect([CUDA_GDB_EXPECT,CUDA_SYN_EXPECT,THREAD_CONTINUE_EXPECT,THREAD_CONTINUE_EXPECT_WERIED,CUDA_SYN_EXPECT_2,pexpect.TIMEOUT],timeout=60)
            target = cuda_gdb_p.before
            #logger.info("in stepi "+target)
            if CUDA_SYN_EXPECT in target or CUDA_SYN_EXPECT_2 in target:
                    #logger.info("Hit the barrier!")
                    cuda_gdb_p.sendline(NEXT)
                    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                    break 
            if j == 0:
                if NO_FOCUS in target and "Switching" not in target:
                    #logger.info("CONTINUE THREADS to hit breakpoint again! -1")
                    cuda_gdb_p.sendline(CONTINUE)
                    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                    target = cuda_gdb_p.before
                    #logger.info("target 1 "+target)
                    time.sleep(5)
                if CUDA_SYN_EXPECT in target:
                    #logger.info("Hit the barrier!")
                    cuda_gdb_p.sendline(NEXT)
                    cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                flag_step = 1
            elif j == 1:
                #logger.info("Hit the barrier! - 1")
                cuda_gdb_p.sendline(NEXT)
                cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                flag_step = 1
            elif j == 2:
                cuda_gdb_p.sendline()
                #logger.info("Send enter in stepi 1")
                flag_step = 1
            elif j == 3:
                cuda_gdb_p.sendline()
                #logger.info("Send enter in stepi 2")
            elif j == 4:
                #logger.info("Hit the barrier! - 2")
                cuda_gdb_p.sendline(NEXT)
                cuda_gdb_p.expect(CUDA_GDB_EXPECT)
                flag_step = 1
            else:
                flag_step = 1
        i = -1
        ################################################################
        #fault injection pseudo technique
        if iterator == position:
            #n_steps, n_faults, cuda_gdb_p
            result = fault_injection(random.randint(1, MAX_STEPS), random.randint(1, MAX_FAULTS), cuda_gdb_p)
            cuda_gdb_p.sendline(DELETE_BREAKPOINT)
            cuda_gdb_p.sendline(CONTINUE)
            return result
        #print iterator
           
def main(argv):
    global N_CONTINUES_9x, N_CONTINUES_1x, N_CONTINUES_4x
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
    spamwriter.writerow(['position', 'argument', 'kernel', 'kernel_line', 'n_faults', 'n_steps', 'registers'])    
    
    for i in range(0, times):
        for parameter in configure_hog.parameter:
            #path, position, argument, kernel, kernel_line
            kernel = random.choice(configure_hog.kernel_names)
            size = ''
            if "9x" in parameter:
                size = "9x"
                position = random.randrange(1, N_CONTINUES_9x)
            elif "4x" in parameter:
                size = "4x"
                position = random.randrange(1, N_CONTINUES_4x)
            else:
                size = "1x"
                position = random.randrange(1, N_CONTINUES_1x)
               
            kernel_line = configure_hog.kernel_start_line[kernel]
            ret = count_continues(binary, position, parameter, configure_hog.filename+":"+str(kernel_line))
            spamwriter.writerow((position, size, kernel, kernel_line, ret.n_faults, ret.n_steps, ' '.join(ret.reg_list)))

    csvfile.close()
 

if __name__ == "__main__" :
    main(sys.argv[1:])
