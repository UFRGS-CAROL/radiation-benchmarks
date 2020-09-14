#!/usr/bin/python

import json

sshcmd="sshpass -p qwerty0 ssh carol@mic0 "
commandList = [
# BFS Input read too slow, kernel time takes 0.9s e input read takes dozen of seconds
# 	[sshcmd+"\" /micNfs/codes/bfs/bfs_check 228 /micNfs/codes/bfs/graph16M.txt /micNfs/codes/bfs/gold-graph16M 9999999\"", 1, sshcmd+"\"  killall -9 bfs_check\""],

# Kmeans plain
 	[sshcmd+"\" /micNfs/codes/kmeans/kmeans_check -i /micNfs/codes/kmeans/kdd_cup -o /micNfs/codes/kmeans/gold-kdd -n 228 -l 9999999\"", 1, sshcmd+"\"  killall -9 kmeans_check\""],

# Lulesh plain
 	[sshcmd+"\" /micNfs/codes/lulesh/lulesh_check -s 15 -g /micNfs/codes/lulesh/gold_15\"", 1, sshcmd+"\"  killall -9 lulesh_check\""],

# Mergesort plain
 	[sshcmd+"\" /micNfs/codes/mergesort/merge_check 67108864 228 /micNfs/codes/mergesort/inputsort_134217728 /micNfs/codes/mergesort/gold_67108864 99999999\"", 1, sshcmd+"\"  killall -9 merge_check\""],

# Quicksort plain
 	[sshcmd+"\" /micNfs/codes/quicksort/quick_check 67108864 228 /micNfs/codes/mergesort/inputsort_134217728 /micNfs/codes/quicksort/gold_67108864 99999999\"", 1, sshcmd+"\"  killall -9 quick_check\""],

# NN plain
 	[sshcmd+"\" /micNfs/codes/nn/nn_check  /micNfs/codes/nn/list8192k.txt 10 10 80 /micNfs/codes/nn/gold-list8192k.txt 8388608 99999999\"", 1, sshcmd+"\"  killall -9 \""],

# DGEMM plain
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check 228 1024 32 /micNfs/codes/dgemm/dgemm_a_1024 /micNfs/codes/dgemm/dgemm_b_1024 /micNfs/codes/dgemm/gold_1024_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check\""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check 228 2048 32 /micNfs/codes/dgemm/dgemm_a_2048 /micNfs/codes/dgemm/dgemm_b_2048 /micNfs/codes/dgemm/gold_2048_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check\""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check 228 4096 32 /micNfs/codes/dgemm/dgemm_a_4096 /micNfs/codes/dgemm/dgemm_b_4096 /micNfs/codes/dgemm/gold_4096_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check\""],

# DGEMM hardening 1
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_1 228 1024 32 /micNfs/codes/dgemm/dgemm_a_1024 /micNfs/codes/dgemm/dgemm_b_1024 /micNfs/codes/dgemm/gold_1024_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_1\""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_1 228 2048 32 /micNfs/codes/dgemm/dgemm_a_2048 /micNfs/codes/dgemm/dgemm_b_2048 /micNfs/codes/dgemm/gold_2048_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_1\""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_1 228 4096 32 /micNfs/codes/dgemm/dgemm_a_4096 /micNfs/codes/dgemm/dgemm_b_4096 /micNfs/codes/dgemm/gold_4096_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_1\""],

# DGEMM hardening 2
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_2 228 1024 32 /micNfs/codes/dgemm/dgemm_a_1024 /micNfs/codes/dgemm/dgemm_b_1024 /micNfs/codes/dgemm/gold_1024_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_2\""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_2 228 2048 32 /micNfs/codes/dgemm/dgemm_a_2048 /micNfs/codes/dgemm/dgemm_b_2048 /micNfs/codes/dgemm/gold_2048_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_2\""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_2 228 4096 32 /micNfs/codes/dgemm/dgemm_a_4096 /micNfs/codes/dgemm/dgemm_b_4096 /micNfs/codes/dgemm/gold_4096_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_2\""],

# DGEMM hardening 3
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_3 228 1024 32 /micNfs/codes/dgemm/dgemm_a_1024 /micNfs/codes/dgemm/dgemm_b_1024 /micNfs/codes/dgemm/gold_1024_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_3\""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_3 228 2048 32 /micNfs/codes/dgemm/dgemm_a_2048 /micNfs/codes/dgemm/dgemm_b_2048 /micNfs/codes/dgemm/gold_2048_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_3\""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_3 228 4096 32 /micNfs/codes/dgemm/dgemm_a_4096 /micNfs/codes/dgemm/dgemm_b_4096 /micNfs/codes/dgemm/gold_4096_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_3\""],

# DGEMM hardening 4
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_4 228 1024 32 /micNfs/codes/dgemm/dgemm_a_1024 /micNfs/codes/dgemm/dgemm_b_1024 /micNfs/codes/dgemm/gold_1024_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_4\""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_4 228 2048 32 /micNfs/codes/dgemm/dgemm_a_2048 /micNfs/codes/dgemm/dgemm_b_2048 /micNfs/codes/dgemm/gold_2048_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_4\""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_4 228 4096 32 /micNfs/codes/dgemm/dgemm_a_4096 /micNfs/codes/dgemm/dgemm_b_4096 /micNfs/codes/dgemm/gold_4096_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_4\""],

# DGEMM hardening 5
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_5 228 1024 32 /micNfs/codes/dgemm/dgemm_a_1024 /micNfs/codes/dgemm/dgemm_b_1024 /micNfs/codes/dgemm/gold_1024_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_5 \""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_5 228 2048 32 /micNfs/codes/dgemm/dgemm_a_2048 /micNfs/codes/dgemm/dgemm_b_2048 /micNfs/codes/dgemm/gold_2048_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_5 \""],
 	[sshcmd+"\" /micNfs/codes/dgemm/dgemm_check_hardened_5 228 4096 32 /micNfs/codes/dgemm/dgemm_a_4096 /micNfs/codes/dgemm/dgemm_b_4096 /micNfs/codes/dgemm/gold_4096_m-order_228_ths_32_blocks 10000000\"", 1, sshcmd+"\"  killall -9 dgemm_check_hardened_5 \""],

# LavaMD plain
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check 228 8 /micNfs/codes/lavamd/input_distance_228_8 /micNfs/codes/lavamd/input_charges_228_8 /micNfs/codes/lavamd/output_gold_228_8 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check\""],
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check 228 9 /micNfs/codes/lavamd/input_distance_228_9 /micNfs/codes/lavamd/input_charges_228_9 /micNfs/codes/lavamd/output_gold_228_9 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check\""],
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check 228 10 /micNfs/codes/lavamd/input_distance_228_10 /micNfs/codes/lavamd/input_charges_228_10 /micNfs/codes/lavamd/output_gold_228_10 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check\""],

# LavaMD hardening 1
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check_hardened_1 228 8 /micNfs/codes/lavamd/input_distance_228_8 /micNfs/codes/lavamd/input_charges_228_8 /micNfs/codes/lavamd/output_gold_228_8 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check_hardened_1\""],
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check_hardened_1 228 9 /micNfs/codes/lavamd/input_distance_228_9 /micNfs/codes/lavamd/input_charges_228_9 /micNfs/codes/lavamd/output_gold_228_9 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check_hardened_1\""],
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check_hardened_1 228 10 /micNfs/codes/lavamd/input_distance_228_10 /micNfs/codes/lavamd/input_charges_228_10 /micNfs/codes/lavamd/output_gold_228_10 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check_hardened_1 \""],

# LavaMD hardening 2
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check_hardened_2 228 8 /micNfs/codes/lavamd/input_distance_228_8 /micNfs/codes/lavamd/input_charges_228_8 /micNfs/codes/lavamd/output_gold_228_8 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check_hardened_2 \""],
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check_hardened_2 228 9 /micNfs/codes/lavamd/input_distance_228_9 /micNfs/codes/lavamd/input_charges_228_9 /micNfs/codes/lavamd/output_gold_228_9 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check_hardened_2 \""],
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check_hardened_2 228 10 /micNfs/codes/lavamd/input_distance_228_10 /micNfs/codes/lavamd/input_charges_228_10 /micNfs/codes/lavamd/output_gold_228_10 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check_hardened_2 \""],

# LavaMD hardening 3
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check_hardened_3 228 8 /micNfs/codes/lavamd/input_distance_228_8 /micNfs/codes/lavamd/input_charges_228_8 /micNfs/codes/lavamd/output_gold_228_8 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check_hardened_3 \""],
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check_hardened_3 228 9 /micNfs/codes/lavamd/input_distance_228_9 /micNfs/codes/lavamd/input_charges_228_9 /micNfs/codes/lavamd/output_gold_228_9 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check_hardened_3 \""],
 	[sshcmd+"\" /micNfs/codes/lavamd/lavamd_check_hardened_3 228 10 /micNfs/codes/lavamd/input_distance_228_10 /micNfs/codes/lavamd/input_charges_228_10 /micNfs/codes/lavamd/output_gold_228_10 10000000\"", 1, sshcmd+"\"  killall -9 lavamd_check_hardened_3 \""],

# LUD plain
 	[sshcmd+"\" /micNfs/codes/lud/lud_check -n 228 -s 1024 -i /micNfs/codes/lud/input_1024_th_228 -g /micNfs/codes/lud/gold_1024_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check \""],
 	[sshcmd+"\" /micNfs/codes/lud/lud_check -n 228 -s 2048 -i /micNfs/codes/lud/input_2048_th_228 -g /micNfs/codes/lud/gold_2048_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check \""],
 	[sshcmd+"\" /micNfs/codes/lud/lud_check -n 228 -s 4096 -i /micNfs/codes/lud/input_4096_th_228 -g /micNfs/codes/lud/gold_4096_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check \""],
# LUD hardening 1
 	[sshcmd+"\" /micNfs/codes/lud/lud_check_hardened_1 -n 228 -s 1024 -i /micNfs/codes/lud/input_1024_th_228 -g /micNfs/codes/lud/gold_1024_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check_hardened_1 \""],
 	[sshcmd+"\" /micNfs/codes/lud/lud_check_hardened_1 -n 228 -s 2048 -i /micNfs/codes/lud/input_2048_th_228 -g /micNfs/codes/lud/gold_2048_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check_hardened_1 \""],
 	[sshcmd+"\" /micNfs/codes/lud/lud_check_hardened_1 -n 228 -s 4096 -i /micNfs/codes/lud/input_4096_th_228 -g /micNfs/codes/lud/gold_4096_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check_hardened_1 \""],
# LUD hardening 2
 	[sshcmd+"\" /micNfs/codes/lud/lud_check_hardened_2 -n 228 -s 1024 -i /micNfs/codes/lud/input_1024_th_228 -g /micNfs/codes/lud/gold_1024_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check_hardened_2 \""],
 	[sshcmd+"\" /micNfs/codes/lud/lud_check_hardened_2 -n 228 -s 2048 -i /micNfs/codes/lud/input_2048_th_228 -g /micNfs/codes/lud/gold_2048_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check_hardened_2 \""],
 	[sshcmd+"\" /micNfs/codes/lud/lud_check_hardened_2 -n 228 -s 4096 -i /micNfs/codes/lud/input_4096_th_228 -g /micNfs/codes/lud/gold_4096_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check_hardened_2 \""],
# LUD hardening 3
 	[sshcmd+"\" /micNfs/codes/lud/lud_check_hardened_3 -n 228 -s 1024 -i /micNfs/codes/lud/input_1024_th_228 -g /micNfs/codes/lud/gold_1024_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check_hardened_3 \""],
 	[sshcmd+"\" /micNfs/codes/lud/lud_check_hardened_3 -n 228 -s 2048 -i /micNfs/codes/lud/input_2048_th_228 -g /micNfs/codes/lud/gold_2048_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check_hardened_3 \""],
 	[sshcmd+"\" /micNfs/codes/lud/lud_check_hardened_3 -n 228 -s 4096 -i /micNfs/codes/lud/input_4096_th_228 -g /micNfs/codes/lud/gold_4096_th_228 -l 10000000\"", 1, sshcmd+"\"  killall -9 lud_check_hardened_3 \""],
]

commands = list()
for c in commandList:
    commands.append({"exec":c[0], "killcmd":c[2]})

json.dump(commands, open("out.json","w"),indent=4)
print "In file 'out.json'"

