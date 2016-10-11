#!/usr/bin/env python

import sys
import math
import csv
import re
from datetime import timedelta, datetime
import os
from glob import glob

def parseSORTS(fi):
	sdc = 0
	end = 0
	abort = 0
	acc_time = 0
	acc_err = 0
	header = "unknown"
	balance = 0
	parsed_errors = 0
	balance_mismatches = 0
	err_counters = [0, 0, 0, 0, 0] # outoforder, corrupted, link error, sync/analisys error
	it_err_counters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # outoforder, corrupted, link error, sync/analisys error, ooo and corrupted, sync and corrupted, sync and ooo, link and corruption, link and ooo, link and sync, 3+ combinations
	it_flags = [0, 0, 0, 0]
	inf_flag = 0

	print("file:",fi)
	print("Counting lines...", end="\t\t\t\t\t\r")
	
	#abre o arquivo
	lines = open(fi, "r")
	
	# percorre todas as linhas do arquivo
	num_lines = sum(1 for line in lines)
	lines.seek(0)
	n = 0
	
	print("Parsing ", num_lines, " lines...", end="\t\t\t\t\t\r")

	for line in lines:
		n+=1
		if (n%100)==0: # status para verificar se o parser engasgou
			print("Parsing... ",int((n*100)/num_lines),"%", end='\t\t\t\t\t\r')

		#extrai informacoes usando expressao regular nas linhas
		m = re.match(".*HEADER size\:(\d*).*", line)
		if m:
			header = m.group(1)
			header.replace(",","-")
		
		m = re.match(".*INF.*", line)
		if m:
			if inf_flag == 0:
				inf_flag = 1
				it_err_counters[11] += 1
			err_counters[4] += 1
		
		m = re.match(".*#IT.*", line)
		if m:
			inf_flag = 0
				
		
		m = re.match(".*SDC.*", line)
		if m: # ocorre o SDC no log apos todos os erros da execucao terem sido printados no log
			inf_flag = 0
			sdc += 1
			errors = []
			if balance != 0:
				print(">>>Warning: Balance is wrong:",balance)
				balance_mismatches += 1
			balance = 0

			err_type_count = 0
			for flag in it_flags:
				if flag != 0:
					err_type_count += 1
			if err_type_count >= 3:
				it_err_counters[10] +=1 # more than 3 types of errors
				for f in range(len(it_flags)):
					it_flags[f] = 0

			if it_flags[0] and it_flags[1]:
				it_err_counters[4] += 1 # ooo and corrupted
				it_flags[0] = 0
				it_flags[1] = 0
			if it_flags[0] and it_flags[3]:
				it_err_counters[5] += 1 # sync and corrupted
				it_flags[0] = 0
				it_flags[3] = 0
			if it_flags[1] and it_flags[3]:
				it_err_counters[6] += 1 # sync and ooo
				it_flags[1] = 0
				it_flags[3] = 0
			if it_flags[2] and it_flags[0]:
				it_err_counters[7] += 1 # link and ooo
				it_flags[2] = 0
				it_flags[0] = 0
			if it_flags[2] and it_flags[1]:
				it_err_counters[8] += 1 # link and corrupted
				it_flags[2] = 0
				it_flags[1] = 0
			if it_flags[2] and it_flags[3]:
				it_err_counters[9] += 1 # link and sync
				it_flags[2] = 0
				it_flags[3] = 0
			for f in range(len(it_flags)):
				if it_flags[f]:
					it_err_counters[f] += 1
				it_flags[f] = 0

		m = re.match(".*AccTime:(\d+.\d+)", line)
		if m:
			acc_time = float(m.group(1))

		m = re.match(".*AccErr:(\d+)", line)
		if m:
			acc_err = int(m.group(1))

		m = re.match(".*ABORT.*", line)
		if m:
			abort = 1

		m = re.match(".*END.*", line)
		if m:
			end = 1

		m = re.match(".*ERR.*\Elements not ordered.*index=(\d+) ([0-9\-]+)\>([0-9\-]+)", line)
		if m:
			err_counters[0] += 1
			it_flags[0] = 1
			parsed_errors += 1

		m = re.match(".*ERR.*\The histogram from element ([0-9\-]+) differs.*srcHist=(\d+) dstHist=(\d+)", line)
		if m:
			if (int(m.group(2)) >= 32) or (((int(m.group(3))-int(m.group(2))) >= 32) and (((int(m.group(3))-int(m.group(2))) % 2) == 0)):
				err_counters[3] += 1
				it_flags[3] = 1
				parsed_errors += 1
				#print(">>>>Warning: Ignoring element corruption - Element: ", m.group(1), "srcHist: ", m.group(2), "dstHist: ", m.group(3))
			else:
				err_counters[1] += 1
				it_flags[1] = 1
				parsed_errors += 1
				balance += int(m.group(2)) - int(m.group(3))
			
		#ERR The link between Val and Key arrays in incorrect. index=2090080 wrong_key=133787990 val=54684 correct_key_pointed_by_val=-1979613866
		m = re.match(".*ERR.*\The link between Val and Key arrays in incorrect.*", line)
		if m:
			err_counters[2] += 1
			it_flags[2] = 1
			parsed_errors += 1
		
	lines.close()

	#if not end:
		#print("Unexpected EOF reached.")
	
	#threadLock.acquire()
	if header != "unknown":
		# salvo algumas informacoes de cada log num arquivo
		filename = './'+benchmark+'_'+header+'/logs_parsed_'+machine_name+'.csv'
		if not os.path.exists(os.path.dirname(filename)):
			try:
				os.makedirs(os.path.dirname(filename))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
				    raise
		flag=0
		if not os.path.exists(filename):
			flag=1
		fp = open(filename, 'a')
		# arquivo csv, separado por "," para abrir num excel da vida
		if flag==1: # cabecalho
			print("Timestamp,Machine,Benchmark,Header,SDC,LOGGED_ERRORS,ACC_ERR,ACC_TIME,ERR_OUTOFORDER,ERR_CORRUPTED,ERR_LINK,ERR_SYNC,IT_OOO,IT_CORRUPTED,IT_LINK,IT_SYNC,IT_OOO_CORR,IT_SYNC_CORR,IT_SYNC_OOO,IT_LINK_OOO,IT_LINK_CORR,IT_LINK_SYNC,IT_MULTIPLE,BALANCE_MISMATCHES,HARD_DETEC,IT_HARD_DETEC,Logname", file=fp)
		print(startDT.ctime(),",",machine_name,",",benchmark,",",header,",",sdc,",",parsed_errors,",",acc_err,",",acc_time,",",err_counters[0],",",err_counters[1],",",err_counters[2],",",err_counters[3],",", end='', file=fp)
		for i in range(len(it_err_counters)-1):
			print(it_err_counters[i],",", end='', file=fp)
		print(balance_mismatches,",",err_counters[4],",",it_err_counters[11],",",fi, file=fp)

		fp.close()
	#threadLock.release()
	print("Done.", end="\t\t\t\t\t\r")
################ => parseSORTS()

def __join(x, y):
	#print(x)
	return os.path.join(x, y)

######### main
# pega todos os arquivos .log na pasta onde esta sendo 
# executado, e nas subpastas tambem
print("Retrieving file list...", end='\r')
all_logs = [y for x in os.walk(".") for y in glob(__join(x[0], '*.log'))]

# vai ordenar por data, "pelo nome do arquivo que eh uma data"
all_logs.sort()

i=0
end=len(all_logs)
prev=0

print("Searching...", end='\t\t\t\r')
# percorre todos os arquivos .log
for fi in all_logs:

	# verifica se o arquivo eh um arquivo de log dos nossos
	m = re.match(".*/(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(.*)_(.*).log", fi.replace("\\", "/"))
	if m:
		year = int(m.group(1))
		month = int(m.group(2))
		day = int(m.group(3))
		hour = int(m.group(4))
		minute = int(m.group(5))
		sec = int(m.group(6))
		benchmark = m.group(7)
		machine_name = m.group(8)

		#recupera a data pelo nome do arquivo
		startDT = datetime(year, month, day, hour, minute, sec)
	
		if (benchmark.find("Sort")!=-1): # pega sorts
			parseSORTS(fi.replace("\\", "/"))

		# adicionar aqui demais parsers


sys.exit(0)
