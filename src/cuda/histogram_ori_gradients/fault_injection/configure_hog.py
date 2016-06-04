#old things
filename = "hog.cu"
benchmark = "hog_without_log"

path="/home/carol/Fernando/radiation-benchmarks/src/cuda/histogram_ori_gradients/fault_injection/"

#binary path will vary 
binary_path = path + "hog_without" # where the binary executable of the application is located

#configuration for launching the benchmark
parameter = "  /home/carol/Fernando/radiation-benchmarks/data/histogram_ori_gradients/1x_pedestrians.jpg --dst_data " + path + "GOLD_1x.data --iterations 1"

#Kernel line by each HOG kernel
kernel_start_line = {}
kernel_start_line["compute_gradients_8UC4_kernel"]      = [569, 0] #first is start line, second is the breakpoint number
kernel_start_line["compute_hists_kernel_many_blocks"]   = [141, 0]
kernel_start_line["normalize_hists_kernel_many_blocks"] = [281, 0]
kernel_start_line["resize_for_hog_kernel"]              = [769, 0]
kernel_start_line["classify_hists_kernel_many_blocks"]  = [409, 0]

#start_compute_gradients_8UC4_kernel = "hog.cu:"+str(kernel_start_line["compute_gradients_8UC4_kernel"])

#kernel names
kernel_names = ["compute_gradients_8UC4_kernel","compute_hists_kernel_many_blocks", "normalize_hists_kernel_many_blocks", "resize_for_hog_kernel", "classify_hists_kernel_many_blocks"]

#critical vars
critical_vars = {}
critical_vars["compute_gradients_8UC4_kernel"]      = ["x", "val.x", "val.y", "val.z", "height", "width","angle_scale", "*(row"]
critical_vars["compute_hists_kernel_many_blocks"]   = ["cell_x", "cell_thread_x", "cell_y", "block_x", "img_block_width", "scale"]
critical_vars["normalize_hists_kernel_many_blocks"] = ["elem", "block_hist_size", "img_block_width", "*(block_hists", "threshold", "*(squares"]
critical_vars["resize_for_hog_kernel"]              = ["x", "y", "sx", "sy", "colOfs"]
critical_vars["classify_hists_kernel_many_blocks"]  = ["product", "win_x", "img_win_width", "img_block_width", "win_block_stride_x", "win_block_stride_y", "*(block_hists", "*(coefs", "free_coef", "threshold", "products"]

def return_next_kernel(previous_kernel):
    k = False
    for i in kernel_names:
        if previous_kernel not in i:
            k = True
            continue
        elif k == True:
            return i
            
