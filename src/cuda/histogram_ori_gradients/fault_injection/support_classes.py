#support classes
import os
import time

class ReturnObj(object):
    faults = {}
    def __init__(self, position, var, set_value):
        self.faults["var"] = var
        self.faults["set_val"] = set_value
        self.faults["old_value"] = ""
        self.faults["set_variable_string"] = ""
        if '(' in var:
            self.faults["var_is_array"] = "yes"
            self.faults["position"] = position
        else:
            self.faults["var_is_array"] = "no"
            self.faults["position"] = -1
    
    def set_variable(self, string_to_send):
        self.faults["set_variable_string"] = string_to_send

class SupportFile(object):
    gdb_string =""
    gdb_file_name = ""
    gdb_file_path = ""
    bin_path = ""
    bin_file = ""
    log_name = ""
    def __init__(self, bin_path, bin_file, kernel_break, parameter, position, set_variable, print_var):
        self.gdb_string = "set pagination off\nfile "+ bin_path + bin_file + "\nbreak "+ kernel_break +"\nignore 1 "+ str(position) + "\nrun "+ parameter + "\ncontinue\n" + set_variable + "\n" + print_var + "\ndelete breakpoint 1\ncontinue\nquit\n"
        self.bin_path = bin_path
        self.bin_file = bin_file
    
    def generate_gdb_file(self, path, file_name):
        new_gdb_file = open(path + file_name, "w")
        new_gdb_file.write(self.gdb_string)
        new_gdb_file.close()
        self.gdb_file_name = file_name
        self.gdb_file_path = path
    
    def run(self, log_name):
        os.system("sudo killall -9 cudbgprocess > /dev/null 2>&1")
        os.system("sudo killall -9 "+ self.bin_file + " > /dev/null 2>&1")
        os.system("sudo killall -9 cuda_gdb > /dev/null 2>&1")
        time.sleep(1)
        os.system("/usr/local/cuda/bin/cuda-gdb -x " + self.gdb_file_path + self.gdb_file_name +" > "+ self.gdb_file_path + log_name)
        self.log_name = log_name

    def get_printed_string(self):
        fault_free_file = open(self.gdb_file_path + self.log_name)
        for line in fault_free_file.readlines():
           if '$1' in line:
                fault_free_file.close()
                return line

        
