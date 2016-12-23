#!/bin/bash

# LANSCE Dec 2016 test, distance data
#Device; Distance cm; Distance Factors -> (20^2)/((x+20)^2) x is the distance in meters
#Sean 1st 706;	60 cm;	0.9425959091
#Sean 2nd 702;	89 cm;	0.9166068772
#K402;		105 cm;	0.9027256673
#TX1C;		126 cm;	0.8849800304
#Xeon Phi;	156 cm;	0.8605229915
#TX1B;		179 cm;	0.8424527084
#TitanX;		197 cm;	0.8287048441
#Sean 3rd 706;	223 cm;	0.8094332975
#Sean 4th 702;	241 cm;	0.7964825737
rm -f summaries.csv
./calcCrossSection.py lansce_neutron.log logs_parsed_carol3-mic0.csv 0.7584183859 
./calcCrossSection.py lansce_neutron.log logs_parsed_carolapu1.csv 0.7739825432
./calcCrossSection.py lansce_neutron.log logs_parsed_carolhsa2.csv 0.8353503053
./calcCrossSection.py lansce_neutron.log logs_parsed_carolk402.csv 0.8179091079
./calcCrossSection.py lansce_neutron.log logs_parsed_carolxeon1.csv 0.7584183859
./calcCrossSection.py lansce_neutron.log logs_parsed_carolxeon2.csv 0.697616106
