#script with cuda-gdb
#author:Fernando Fernandes

#dizable for return or continue messages
set pagination off

#load the hog_bin
file hog_without


break hog.cu:556

define jump_to
    set $i = 0
    while ($i++ < $arg0)
        next
    end
end

########################################################################
#when gdb is at line 553
define fi_compute_gradients_8UC4_kernel
    jump_to 3
    set variable x = width_in
    #parameters
    set variable angle_scale = 1.3 
end

########################################################################
#when gdb is at line 756 or 765
define fi_resize_for_hog_kernel
    set variable x = 0
    set variable y = 0
end

########################################################################
#when gdb is at line 117
define fi_compute_hists_kernel_many_blocks 
        jump_to 12
        set variable cell_thread_x = 13

        jump_to 3
        set variable bin_id = 1000000
        
        jump_to 4
        set variable dist_y = dist_y_begin + 12     
end

define fi_normalize_hists_kernel_many_block

end

run /home/carol/Fernando/GPU-Injector/hog_without_opencv/4x_pedestrians.jpg --dst_data GOLD_4x.data --hit_threshold 0.9 --gr_threshold 1 --nlevels 100
set $i = 0

cuda block (2,0,0) thread (64,0,0)
while(1)
   set $i++  
   print $i
   continue
end

print $arg1

delete breakpoints
continue
q
