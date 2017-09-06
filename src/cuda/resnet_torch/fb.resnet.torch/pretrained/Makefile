
#./classifi_radiation.lua [model] [radiation mode generate/rad_test] [txt dataset] [gold_file] [iterations -- ignored for gen]

generate: create_dir
	./classify_radiation.lua /home/carol/radiation-benchmarks/data/resnet_torch/resnet-200.t7 generate \
			/home/carol/radiation-benchmarks/data/networks_img_list/fault_injection.txt \
			/home/carol/radiation-benchmarks/data/resnet_torch/fault_injection_gold.csv \
			1
			
test:
	./classify_radiation.lua /home/carol/radiation-benchmarks/data/resnet_torch/resnet-200.t7 rad_test \
		/home/carol/radiation-benchmarks/data/networks_img_list/fault_injection.txt \
		/home/carol/radiation-benchmarks/data/resnet_torch/fault_injection_gold.csv \
		1

gen_rad: create_dir
	./classify_radiation.lua /home/carol/radiation-benchmarks/data/resnet_torch/resnet-200.t7 generate \
				/home/carol/radiation-benchmarks/data/networks_img_list/urban.street.1.1K.txt \
				/home/carol/radiation-benchmarks/data/resnet_torch/urban_test_gold.csv 1	

test_rad:
	./classify_radiation.lua /home/carol/radiation-benchmarks/data/resnet_torch/resnet-200.t7  rad_test \
				 /home/carol/radiation-benchmarks/data/networks_img_list/urban.street.1.1K.txt \
				 /home/carol/radiation-benchmarks/data/resnet_torch/urban_test_gold.csv 1000 log

create_dir:
	mkdir -p ~/radiation-benchmarks/data/resnet_torch

