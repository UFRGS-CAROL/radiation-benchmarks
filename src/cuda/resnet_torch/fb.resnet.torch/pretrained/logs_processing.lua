-- this file will have the code for processing all logs functions
-- LogHelper lua wraper must be compiled before using this script


local lh = require '../../../../include/log_helper_swig_wraper/log_helper'

-- start a log file
function start_app(network, dataset, gold_file, log)
  -- from C start_log_file(char *benchmark_name, char *test_info);
  if log then

    local benchmark = "cudaResnet" .. network

    local header = "img_list_txt: " .. dataset .. " gold_file: " .. gold_file
    lh.start_log_file(benchmark, header)
  end
end

function end_app(log)
  -- from C  end_log_file();
  if log then
    lh.end_log_file()
  end
end


function save_gold(gold, datapath)
  	
end
