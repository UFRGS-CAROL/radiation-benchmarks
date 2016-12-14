#!/bin/bash

rm -f summaries.csv
./calcCrossSection.py lansce_neutron.log logs_parsed_carol3-mic0.csv 0.7584183859 
./calcCrossSection.py lansce_neutron.log logs_parsed_carolapu1.csv 0.7739825432
./calcCrossSection.py lansce_neutron.log logs_parsed_carolhsa2.csv 0.8353503053
./calcCrossSection.py lansce_neutron.log logs_parsed_carolk402.csv 0.8179091079
./calcCrossSection.py lansce_neutron.log logs_parsed_carolxeon1.csv 0.7584183859
./calcCrossSection.py lansce_neutron.log logs_parsed_carolxeon2.csv 0.697616106
