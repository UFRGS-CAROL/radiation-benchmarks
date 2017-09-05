-- this file will have the code for processing all logs functions
-- LogHelper lua wraper must be compiled before using this script


local lh = require '../../../../include/log_helper_swig_wraper/log_helper'
local THRESHOLD = 0.005
-- start a log file
function start_app(network, dataset, gold_file, log)
  -- from C start_log_file(char *benchmark_name, char *test_info);
  if log then

    local benchmark = "cudaResnet"

    local header = "model: " .. network .. " img_list_txt: " .. dataset .. " gold_file: " .. gold_file
    lh.start_log_file(benchmark, header)
  end
end

function end_app(log)
  -- from C  end_log_file();
  if log then
    lh.end_log_file()
  end
end

function start_iteration(log)
  if log then
    lh.start_iteration()
  end
end

function end_iteration(log)
  if log then
    lh.end_iteration()
  end
end

-- save gold for an image, in an opened file
-- gold_file
function save_gold(gold_file, img, probs, classes, tensor_size)
  -- tensor size, img, #probs, #classes
  gold_file:write(tensor_size .. ";" .. img .. ";" .. #(probs:storage()) .. ";" .. #(classes:storage()) .. ";\n")
  for n=1,tensor_size do
    -- probs[n];indexes[n]
    gold_file:write(probs[n] .. ";" .. classes[n] .. ";\n")
  end
end

-- load gold
function load_gold(gold_path)
  local gold_content = {}
  local gold_file = io.open(gold_path, "r")

  if gold_file ~= nil then

    -- read first the size of the dataset
    local line = gold_file:read()
    local splited = split(line, ";")
    local dataset_size = tonumber(splited[0])

    for j = 0,dataset_size do
      line = gold_file:read()
      if line == nil then
        break
      end
      local probs = {}
      local classes = {}
      -- read first header
      splited = split(line, ";")
      local tensor_size = tonumber(splited[0])
      local img = splited[1]
      local probs_size = tonumber(splited[2])
      local classes_size = tonumber(splited[3])

      -- read each probability
      for i = 1, probs_size do
        line = gold_file:read()
        splited = split(line, ";")
        --probs[n];indexes
        probs[i] = tonumber(splited[0])
        classes[i] = tonumber(splited[1])

      end

      gold_content[img] = {probs, classes}
    end
    gold_file:close()
  else
    print("Gold file not open", gold_path)
    return nil
  end

  return gold_content
end

-- compare and log if log is activated
function compare_and_log(log, tensor_size, gold_probs, gold_indexes, found_probs, found_indexes, img, iteration)
  local error_count = 0

  for i=1,tensor_size do
    local gp = gold_probs[i]
    local gi = gold_indexes[i]
    local fp = found_probs[i]
    local fi = found_indexes[i]
    local diff_probs = math.abs(gp - fp)
    local diff_indexes = math.abs(gi - fi)

    -- compare and log if it is greater than threashold
    if diff_probs > THRESHOLD or diff_indexes > THRESHOLD then
      local error_string = string.format("img: [%s] iteration: [%d] found_prob: [%f] gold_prob: [%f] found_index: [%d] gold_index: [%d]",
        img, iteration, fp, gp, fi, gi)
      if log then
        lh.log_error_detail(error_string)
      else
        print(error_string)
      end
      error_count = error_count + 1
    end

  end

  if log then
    lh.log_error_count(error_count)
  end

end

-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end


-- get all lines from a file, returns an empty
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  local lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end


function split(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=0
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end


