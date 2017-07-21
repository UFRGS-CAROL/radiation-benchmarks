import re
import sys

dollar = re.escape('$')
space = re.escape(' ')
decl_symbol = re.escape('%')

keywords = ['while', 'if', 'else', '//', '/*', '*/']
type_dict = dict()

def patch_code(code_file):
        with open (code_file, "r") as myfile:
                data=myfile.readlines()
        
                buf_str = ''

                buf_str += make_hardening_header()
                
                for line in data:        
                        parts = re.split('\t|'+space+'|'+decl_symbol+'|;', line)
                        parts = list(filter(None, parts))
                        #for s in parts:
                                #print(s)
                        #print(parts)
                        if '%' in line:
                                #print(parts)
                                buf_str += make_decl(parts[0], parts[1])
                                
			elif 'for' in parts:
				ignore_step = False
				for keyword in keywords:
                                        if keyword in line or 'omp' in line:
                                                ignore_step = True

                                if ignore_step is True:
                                        buf_str += line
                                        continue

				buf_str += line
				print(parts)
                        elif '=' in parts:
                                ignore_step = False
                                for keyword in keywords:
                                        if keyword in line and 'omp' not in line:
                                                ignore_step = True

                                if ignore_step is True:
                                        buf_str += line
                                        continue
                                equals_index = parts.index('=')
                                #print(parts)
                                #print(equals_index)
                                if '$' in ''.join(parts[:equals_index]):
                                        #print(parts)
                                        var_name = re.split(dollar, parts[0])[1]
                                        
                                        del parts[0]
                                        del parts[0]
                                        parts = parts[:-1] + [';\n']
                                        #print(parts)
                                        buf_str += make_attrib(var_name,''.join(parts), True)

                                else:
                                        var_name = parts[0]
                                        #print(parts)
                                        del parts[0]
                                        del parts[0]
                                        #print(parts)
                                        parts = parts + [';\n']
                                        #print(parts)
                                        buf_str += make_attrib(var_name,''.join(parts), False)

                        elif '$' in line:
                                buf_str += check_read_var(line)
                                
                        elif '@' in line:
                                buf_str += check_read_openmp(line)
                                
                        else:
                                buf_str += line
                                
        return buf_str

def make_decl(var_type, var_name):
        #print ('decl:')

        decl_str = ''
        #print ('{} {}_hardened_1;'.format(var_type, var_name))
        decl_str += '{} {}_hardened_1;\n'.format(var_type, var_name)
        
        #print ('{} {}_hardened_2;'.format(var_type, var_name))
        decl_str += '{} {}_hardened_2;\n'.format(var_type, var_name)
        
        type_dict[var_name] = var_type

        return decl_str
        
def make_attrib(var_name, var_value, is_hardened):
        #print ('attrib:')

        attrib_str = ''
        if is_hardened is True:
                #print ('{}_hardened_1 = {}'.format(var_name, check_read_var(var_value)))
                attrib_str += '{}_hardened_1 = {}'.format(var_name, check_read_var(var_value))
                #print ('{}_hardened_2 = {}'.format(var_name, check_read_var(var_value)))
                attrib_str += '{}_hardened_2 = {}'.format(var_name, check_read_var(var_value))

        else:
                #print ('{} = {}'.format(var_name, check_read_var(var_value)))
                attrib_str += '{} = {}'.format(var_name, check_read_var(var_value))
                
        return attrib_str

def check_read_var(statement):
        parts = re.split('(\W)', statement)

        new_statement = ''
        position = 0
        must_skip = False
        for s in parts:
                if s is '$':
                        if must_skip is False:
                                var_name = parts[position+1]
                                var_type = type_dict[var_name]
                                new_statement += make_read(var_name, var_type)
                                
                                must_skip = True

                        else:
                                must_skip = False
                                
                elif must_skip is False:
                        new_statement += s
                        
                position += 1

        return new_statement

def check_read_openmp(statement):
        parts = re.split('(\W)', statement)
        new_statement = ''
        position = 0
        must_skip = False
        for s in parts:
                if s is '@':
                        if must_skip is False:
                                var_name = parts[position+1]
                                
                                new_statement += '{}_hardened_1, {}_hardened_2'.format(var_name, var_name)
                                
                                must_skip = True

                        else:
                                must_skip = False
                                
                elif must_skip is False:
                        new_statement += s
                        
                position += 1

        return new_statement

def make_read(var_name, var_type):
        read = 'READ_HARDENED_VAR({}_hardened_1, {}_hardened_2, {}, sizeof({}))'.format(var_name, var_name, var_type, var_type)
        return read

def make_hardening_header():
        header_str = ''
        header_str += '#define READ_HARDENED_VAR(VAR_NAME_1, VAR_NAME_2, VAR_TYPE, VAR_SIZE) (*((VAR_TYPE*)hardened_compare_and_return((void*)(&VAR_NAME_1), (void*)(&VAR_NAME_2), VAR_SIZE)))\n'
        header_str += '#define READ_HARDENED_ARRAY(ARRAY_NAME_1, ARRAY_NAME_2, ARRAY_TYPE, ARRAY_SIZE) ((ARRAY_TYPE)((void*)hardened_compare_and_return_array((void*)(&ARRAY_NAME_1), (void*)(&ARRAY_NAME_2), ARRAY_SIZE)))\n'
        header_str += '#define READ_HARDENED_DOUBLE(VAR_NAME_1, VAR_NAME_2, VAR_TYPE, VAR_SIZE) hardened_compare_and_return_double(VAR_NAME_1, VAR_NAME_2)\n'

        header_str += 'inline void* hardened_compare_and_return(void* var_a, void* var_b, long long size)\n'
        header_str += '{\n'
        header_str += '\tif(memcmp(var_a, var_b, size) != 0)\n'
        header_str += '\t{\n'
        header_str += '\t\tprintf("\\nHardening error: at file \\\"%s\\\"\\n\\n\", __FILE__);\n'
        header_str += '\t\texit(1);\n'
        header_str += '\t}\n'
        
        header_str += '\treturn var_a;\n'
        header_str += '}\n'

        header_str += 'inline void* hardened_compare_and_return_array(void* array_ptr_a, void* array_ptr_b, long long size)\n'
        header_str += '{\n'
        header_str += '\tchar* bytes_array_a = (char*)((char**)array_ptr_a);\n'
        header_str += '\tchar* bytes_array_b = (char*)((char**)array_ptr_b);\n'

        header_str += '\tif(memcmp(bytes_array_a, bytes_array_b, size) != 0)\n'
        header_str += '\t{\n'
        header_str += '\t\tprintf("\\nHardening error: at file \\\"%s\\\"\\n\\n\", __FILE__);\n'
        header_str += '\t\texit(1);\n'
        header_str += '\t}\n'

        header_str += '\treturn array_ptr_a;\n'
        header_str += '}\n'

        return header_str

if len(sys.argv) is not 2:
        print ('Inform the .c file to be hardened!\n')

else:
        input_file = sys.argv[1]
        #print(patch_code(input_file))
        with open('hardened_'+input_file, "w") as output:
                output.write(patch_code(input_file))
