#!/usr/bin/python -u

import os
import sys
import re
import csv

from datetime import timedelta
from datetime import datetime

def getDt(yearDate, dayTime, secFrac):
    yv = yearDate.split('/')
    day = int(yv[0])
    month = int(yv[1])
    year = int(yv[2])

    dv = dayTime.split(':')
    hour = int(dv[0])
    minute = int(dv[1])
    second = int(dv[2])

    # we get secFrac in seconds, so we must convert to microseconds
    # e.g: 0.100 seconds = 100 milliseconds = 100000 microseconds
    microsecond = int(float(secFrac) * 1e6)

    return datetime(year, month, day, hour, minute, second, microsecond)

fileLines = list()
def readCountFile():
	inFile = open(inFileName, 'r')
        global fileLines
	for l in inFile:

		# Sanity check, we require a date at the beginning of the line
		line = l.rstrip()
		if not re.match("\d{1,2}/\d{1,2}/\d{2,4}", line):
			#sys.stderr.write("Ignoring line (malformed):\n%s\n" % (line))
			continue
		
		if "N/A" in line:
		    break
		
                fileLines.append(line)

def getFluenceFlux(startDT, endDT):

	#inFile = open(inFileName, 'r')
	#endDT = startDT + timedelta(minutes=60)

        last_counter_20 = 0
        last_counter_30 = 0
        last_counter_40 = 0
        last_curIntegral = 0
	lastDT = None
        flux1h = 0
	beamOffTime = 0
        first_curIntegral = None

	#for l in inFile:
	for l in fileLines:

		## Sanity check, we require a date at the beginning of the line
		line = l.rstrip()
		#if not re.match("\d{1,2}/\d{1,2}/\d{2,4}", line):
		#	#sys.stderr.write("Ignoring line (malformed):\n%s\n" % (line))
		#	continue
		#
		#if "N/A" in line:
		#    break
		#
		## Parse the line
		lv = line.split(';')

		yearDate = lv[0]
		dayTime = lv[1]
		secFrac = lv[2]
		counter_40 = lv[3]
		counter_20 = lv[4]
		counter_30 = lv[5]
		fission_counter = lv[6]
		curIntegral = lv[7]
		current = lv[8]
		
		# Generate datetime for line
		curDt = getDt(yearDate, dayTime, secFrac)

		if startDT <= curDt and first_curIntegral is None:
			first_curIntegral = float(curIntegral)
			last_counter_20 = counter_20
			last_counter_30 = counter_30
			last_counter_40 = counter_40
			lastDT = curDt
			continue

		if first_curIntegral is not None:
			if counter_30 == last_counter_30:
				shutter = "Closed"
				beamOffTime += (curDt - lastDT).total_seconds()
			else:
				shutter = "Open"
			last_counter_20 = counter_20
			last_counter_30 = counter_30
			last_counter_40 = counter_40
			lastDT = curDt

		if curDt > endDT:
			flux1h =  (float(last_curIntegral) - first_curIntegral)/(endDT - startDT).total_seconds()
			return [flux1h, beamOffTime]
                elif first_curIntegral is not None:
                        last_curIntegral = curIntegral

#def getFlux(startDT):
#
#	inFile = open(inFileName, 'r')
#	endDT = startDT + timedelta(minutes=60)
#
#	beamIntegralSum = 0
#	timeBeamOff=0 # in seconds
#
#	for l in inFile:
#
#		# Sanity check, we require a date at the beginning of the line
#		line = l.rstrip()
#		if not re.match("\d{1,2}/\d{1,2}/\d{2,4}", line):
#			#sys.stderr.write("Ignoring line (malformed):\n%s\n" % (line))
#			continue
#		
#		if "N/A" in line:
#		    break
#		
#		# Parse the line
#		lv = line.split(';')
#
#		yearDate = lv[0]
#		dayTime = lv[1]
#		secFrac = lv[2]
#		unknown = lv[3]
#		thermalNeutronsCount = lv[4]
#		fastNeutronsCount = lv[5]
#		efficientCount = lv[6]
#		synchrotronCurrent = lv[7]
#		shutter = lv[8]
#		elapsedTime = lv[9]
#		beamIntegral = lv[10]
#		
#		# Generate datetime for line
#		curDt = getDt(yearDate, dayTime, secFrac)
#		
#		if startDT <= curDt and curDt <= endDT:
#			beamIntegralSum += float(lv[10])
#			if shutter == "Closed" or float(synchrotronCurrent) < 50:
#				timeBeamOff += float(elapsedTime)
#		elif curDt > endDT:
#			flux1h = (beamIntegralSum*factor)/180
#			return [flux1h, timeBeamOff]


#########################################################
#                    Main Thread                        #
#########################################################
if len(sys.argv) < 4:
    print "Usage: %s <neutron counts input file> <csv file> <factor>" % (sys.argv[0])
    sys.exit(1)

inFileName = sys.argv[1]
csvFileName = sys.argv[2]
factor = float(sys.argv[3])

csvOutFileName = csvFileName.replace(".csv", "_cross_section.csv")
csvOutFileName2 = csvFileName.replace(".csv", "_cross_section_summary.csv")

print "in: "+csvFileName
print "out: "+csvOutFileName

csvFP = open(csvFileName, "r")
reader = csv.reader(csvFP, delimiter=';')
csvWFP = open(csvOutFileName, "w")
writer = csv.writer(csvWFP, delimiter=';')
csvWFP2 = open(csvOutFileName2, "w")
writer2 = csv.writer(csvWFP2, delimiter=';')

csvHeader = next(reader, None)


writer.writerow(csvHeader)
writer2.writerow(csvHeader)

lines = list(reader)

# We need to read the neutron count files before calling getFluenceFlux
readCountFile()
i=0
while i < len(lines):
	startDT = datetime.strptime(lines[i][0][0:-1], "%c")
	print "date in line "+str(i)+": ",startDT
	j = i
	accTimeS = float(lines[i][6])
	sdcS = int(lines[i][4])
	abortZeroS = 0
	if(int(lines[i][7]) == 0):
		abortZeroS += 1
	writer.writerow(lines[i])
	writer2.writerow(lines[i])
        endDT = datetime.strptime(lines[i+1][0][0:-1], "%c")
        lastLine = ""
	while((endDT - startDT) < timedelta(minutes=60) ):
		if(lines[i+1][2] != lines[i][2]): # not the same benchmark
			break;
		if(lines[i+1][3] != lines[i][3]): # not the same input
			break;
		#print "line "+str(i)+" inside 1h interval"
		i += 1
		accTimeS += float(lines[i][6])
		sdcS += int(lines[i][4])
		if(int(lines[i][7]) == 0):
			abortZeroS += 1
		writer.writerow(lines[i])
                lastLine = lines[i]
		if i == (len(lines)-1): # end of lines
			break;
                endDT = datetime.strptime(lines[i+1][0][0:-1], "%c")
	#compute 1h flux; sum SDC, ACC_TIME, Abort with 0; compute fluence (flux*(sum ACC_TIME))
	flux,timeBeamOff = getFluenceFlux(startDT,(startDT + timedelta(minutes=60)))
	fluxAccTime,timeBeamOffAccTime = getFluenceFlux(startDT,(startDT + timedelta(seconds=accTimeS)))
	fluence = flux*accTimeS
	fluenceAccTime = fluxAccTime*accTimeS
	if fluence > 0:
		crossSection = sdcS/fluence
		crossSectionCrash = abortZeroS/fluence
	else:
		crossSection = 0
		crossSectionCrash = 0
        if fluenceAccTime > 0:
		crossSectionAccTime = sdcS/fluenceAccTime
		crossSectionCrashAccTime = abortZeroS/fluenceAccTime
        else:
		crossSectionAccTime = 0
                crossSectionCrashAccTime = 0
	headerC = ["start timestamp", "end timestamp", "#lines computed", "#SDC", "#AccTime", "#(Abort==0)"]
	headerC.append("Flux 1h (factor "+str(factor)+")")
	headerC.append("Flux AccTime (factor "+str(factor)+")")
	headerC.append("Fluence(Flux * $AccTime)")
	headerC.append("Fluence AccTime(FluxAccTime * $AccTime)")
	headerC.append("Cross Section SDC")
	headerC.append("Cross Section Crash")
	headerC.append("Time Beam Off (sec)")
	headerC.append("Cross Section SDC AccTime")
	headerC.append("Cross Section Crash AccTime")
	headerC.append("Time Beam Off AccTime (sec)")
	writer.writerow(headerC)
	writer2.writerow(lastLine)
	writer2.writerow(headerC)
	row = [startDT.ctime(), endDT.ctime(), (i-j+1), sdcS, accTimeS, abortZeroS, flux, fluxAccTime, fluence, fluenceAccTime, crossSection, crossSectionCrash, timeBeamOff, crossSectionAccTime, crossSectionCrashAccTime, timeBeamOffAccTime]
	writer.writerow(row)
	writer2.writerow(row)
	writer.writerow([])
	writer.writerow([])
	writer2.writerow([])
	writer2.writerow([])
	i += 1

csvFP.close()
csvWFP.close()
csvWFP2.close()

sys.exit(0)
