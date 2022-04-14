#script with cuda-gdb
#author:Fernando Fernandes

#dizable for return or continue messages
set pagination off

#load the hog_bin
file hog_without


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

define many_faults
    set $per = 11
    while ($per <= 12)
        break App.h:156
        run ../../../../data/histogram_ori_gradients/1x_pedestrians.jpg --dst_data GOLD_1x.data --iterations 1
        next
        next
        step
        next
        set $i = 0 
        set $tam = 1
        print $tam
        while($i <  $tam) 
           set $jump = 700 + $i
           set $i++
           set variable detector[$jump] = detector[$jump] * 1000 
        end
        delete breakpoints
        continue
        if($per == 1)
            shell mkdir /var/radiation-benchmarks/log/1
            shell mv /var/radiation-benchmarks/log/*log /var/radiation-benchmarks/log/1/
        end
        if($per == 2)
            shell mkdir /var/radiation-benchmarks/log/2
            shell mv /var/radiation-benchmarks/log/*log /var/radiation-benchmarks/log/2/
        end
        if($per == 3)
            shell mkdir /var/radiation-benchmarks/log/3
            shell mv /var/radiation-benchmarks/log/*log /var/radiation-benchmarks/log/3/
        end
        if($per == 4)
            shell mkdir /var/radiation-benchmarks/log/4
            shell mv /var/radiation-benchmarks/log/*log /var/radiation-benchmarks/log/4/
        end
        if($per == 5)
            shell mkdir /var/radiation-benchmarks/log/5
            shell mv /var/radiation-benchmarks/log/*log /var/radiation-benchmarks/log/5/
        end
        if($per == 6)
            shell mkdir /var/radiation-benchmarks/log/6
            shell mv /var/radiation-benchmarks/log/*log /var/radiation-benchmarks/log/6/
        end
        if($per == 7)
            shell mkdir /var/radiation-benchmarks/log/7
            shell mv /var/radiation-benchmarks/log/*log /var/radiation-benchmarks/log/7/
        end 
        if($per == 8)
            shell mkdir /var/radiation-benchmarks/log/8
            shell mv /var/radiation-benchmarks/log/*log /var/radiation-benchmarks/log/8/
        end
        if($per == 9)
            shell mkdir /var/radiation-benchmarks/log/9
            shell mv /var/radiation-benchmarks/log/*log /var/radiation-benchmarks/log/9/
        end
        if($per == 10)
            shell mkdir /var/radiation-benchmarks/log/10
            shell mv /var/radiation-benchmarks/log/*log /var/radiation-benchmarks/log/10/
        end
        set $per += 1

    end
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

set $i = 0
while($i < 748)
    break App.h:156
    run ../../../../data/histogram_ori_gradients/1x_pedestrians.jpg --dst_data GOLD_1x.data --iterations 10
    next
    next
    step
    next
    set variable detector[$i] = detector[$i] * 100 
    delete breakpoints
    continue
    set $i++
end
q
