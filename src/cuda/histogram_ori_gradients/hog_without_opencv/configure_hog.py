#old things
filename = "hog.cu"
benchmark = "hog_without"

#binary path will vary 
binary_path = "/home/carol/Fernando/GPU-Injector/hog_without_opencv/hog_without" # where the binary executable of the application is located

#configuration for launching the benchmark
parameter = [" /home/carol/Fernando/GPU-Injector/hog_without_opencv/1x_pedestrians.jpg --dst_data GOLD_1x.data --iterations 1",
             " /home/carol/Fernando/GPU-Injector/hog_without_opencv/4x_pedestrians.jpg --dst_data GOLD_4x.data --iterations 1",
             " /home/carol/Fernando/GPU-Injector/hog_without_opencv/9x_pedestrians.jpg --dst_data GOLD_9x.data --iterations 1"]

#Kernel line by each HOG kernel
kernel_start_line = {}
kernel_start_line["compute_gradients_8UC4_kernel"]      = 556
kernel_start_line["resize_for_hog_kernel"]              = 758
kernel_start_line["resize_for_hog_kernel2"]             = 767
kernel_start_line["compute_hists_kernel_many_blocks"]   = 120
kernel_start_line["normalize_hists_kernel_many_blocks"] = 267

start_compute_gradients_8UC4_kernel = "hog.cu:"+str(kernel_start_line["compute_gradients_8UC4_kernel"])

#kernel names
kernel_names = ["compute_gradients_8UC4_kernel", "resize_for_hog_kernel", "resize_for_hog_kernel2",
                "compute_hists_kernel_many_blocks", "normalize_hists_kernel_many_blocks"]

#kernel registers list
register_list = ["CC", "P0", "P1", "P2", "P3", "P4", "P5", "P6", "R0", "R1", "R2", 
"R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", 
"R14", "R15", "R16", "R17", "R18", "R19", "R20", "R21", "R22", "R23", 
"R24", "R25", "R26", "R27", "R28", "R29", "R30", "R31", "R32", "R33", 
"R34", "R35", "R36", "R37", "R38", "R39", "R40", "R41", "R42", "R43", 
"R44", "R45", "R46", "R47", "R48", "R49", "R50", "R51", "R52", "R53", 
"R54", "R55", "R56", "R57", "R58", "R59", "R60", "R61", "R62", "R63", 
"R64", "R65", "R66", "R67", "R68", "R69", "R70", "R71", "R72", "R73", 
"R74", "R75", "R76", "R77", "R78", "R79", "R80", "R81", "R82", "R83", 
"R84", "R85", "R86", "R87", "R88", "R89", "R90", "R91", "R92", "R93", 
"R94", "R95", "R96", "R97", "R98", "R99", "R100", "R101", "R102", "R103", 
"R104", "R105", "R106", "R107", "R108", "R109", "R110", "R111", "R112", 
"R113", "R114", "R115", "R116", "R117", "R118", "R119", "R120", "R121", 
"R122", "R123", "R124", "R125", "R126", "R127", "R128", "R129", "R130", 
"R131", "R132", "R133", "R134", "R135", "R136", "R137", "R138", "R139", 
"R140", "R141", "R142", "R143", "R144", "R145", "R146", "R147", "R148", 
"R149", "R150", "R151", "R152", "R153", "R154", "R155", "R156", "R157", 
"R158", "R159", "R160", "R161", "R162", "R163", "R164", "R165", "R166", 
"R167", "R168", "R169", "R170", "R171", "R172", "R173", "R174", "R175", 
"R176", "R177", "R178", "R179", "R180", "R181", "R182", "R183", "R184", 
"R185", "R186", "R187", "R188", "R189", "R190", "R191", "R192", "R193", 
"R194", "R195", "R196", "R197", "R198", "R199", "R200", "R201", "R202", 
"R203", "R204", "R205", "R206", "R207", "R208", "R209", "R210", "R211", 
"R212", "R213", "R214", "R215", "R216", "R217", "R218", "R219", "R220", 
"R221", "R222", "R223", "R224", "R225", "R226", "R227", "R228", "R229", 
"R230", "R231", "R232", "R233", "R234", "R235", "R236", "R237", "R238", 
"R239", "R240", "R241", "R242", "R243", "R244", "R245", "R246", "R247", 
"R248", "R249", "R250", "R251", "R252", "R253", "R254", "R255", "R256"]
