#!/home/carol/radiation-benchmarks/src/cuda/resnet_torch/torch/install/bin/th

--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local lp = require 'logs_processing'
local t = require '../datasets/transforms'
local imagenetLabel = require './imagenet'


--/////////////////////////////////////////////////////

function parse_args()
  if #arg < 2 then
    io.stderr:write('Usage for normal execution: ./classify.lua [MODEL] [FILE]...\n')
    io.stderr:write('Usage for radiation execution: ./classifi_radiation.lua [model] \
    [radiation mode generate/rad_test] [txt dataset] [gold_file] [iterations -- ignored for gen] [log]\n')
    os.exit(1)
  end
  for _, f in ipairs(arg) do
    if not paths.filep(f) then
      io.stderr:write('file not found: ' .. f .. '\n')
      os.exit(1)
    end
    break
  end

  -- return the variables
  -- rad mode
  local weights, rad_mode, txt_dataset, gold_file, iterations = nil
  log = false
  if #arg > 3 then
    local i = 1
    weights = arg[i]
    i = i + 1
    rad_mode = arg[i]
    i = i + 1
    txt_dataset = arg[i]
    i = i + 1
    gold_file = arg[i]
    i = i + 1
    iterations = arg[i]
    i = i + 1
    if arg[i] ~= nil and (arg[i] == "1" or string.lower(arg[i]) == "true" or string.lower(arg[i]) == "log") then
      log = true
    end
  else
    for _, f in ipairs(arg) do
      if not paths.filep(f) then
        io.stderr:write('file not found: ' .. f .. '\n')
        os.exit(1)
      end
    end

  end

  return weights, rad_mode, txt_dataset, gold_file, iterations, log
end

function load_model()
  print("Loading Resnet models " .. weights)
  -- Load the model
  local model = torch.load(weights):cuda()
  local softMaxLayer = cudnn.SoftMax():cuda()

  -- add Softmax layer
  model:add(softMaxLayer)

  -- Evaluate mode
  model:evaluate()

  -- The model was trained with this input normalization
  local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
  }

  local transform = t.Compose{
    t.Scale(256),
    t.ColorNormalize(meanstd),
    t.CenterCrop(224),
  }

  print("Models and Network created with sucess")

  return model, softMaxLayer, transform, meanstd
end

function normal_execution()
  local model_obj, softMaxLayer, transform, meanstd = load_model()
  local N = 10

  for i=2,#arg do
    print(i)
    -- load the image as a RGB float tensor with values 0..1
    local img = image.load(arg[i], 3, 'float')
    local name = arg[i]:match( "([^/]+)$" )

    -- Scale, normalize, and crop the image
    img = transform(img)

    -- View as mini-batch of size 1
    local batch = img:view(1, table.unpack(img:size():totable()))

    -- Get the output of the softmax
    local output = model_obj:forward(batch:cuda()):squeeze()

    -- Get the top 5 class indexes and probabilities
    local probs, indexes = output:topk(N, true, true)
    print('Classes for', arg[i])
    for n=1,N do
      print(probs[n], imagenetLabel[indexes[n]])
    end
    print('')

  end

end


function generate_radiation_test(arg_size)
  local model_obj, softMaxLayer, transform, meanstd = load_model()

  local N = 0
  local lines = lines_from(txt_dataset)
  gold_file = io.open(gold_path, "w")
  gold_file:write(#lines .. ";\n")
  if gold_file then
    for k,v in pairs(lines) do

      -- load the image as a RGB float tensor with values 0..1
      -- [model] [radiation mode gen/rad] [txt dataset] [gold_file] [iterations -- ignored for gen]
      local img = image.load(v, 3, 'float')
      local name = v:match( "([^/]+)$" )


      -- Scale, normalize, and crop the image
      img = transform(img)

      -- View as mini-batch of size 1
      local batch = img:view(1, table.unpack(img:size():totable()))

      -- Get the output of the softmax
      local output = model_obj:forward(batch:cuda()):squeeze()

      -- get the dim
      if output:dim() == 1 then
        N = tonumber(#(output:storage()))
      end

      -- Get the all class indexes and probabilities
      local probs, indexes = output:topk(N, true, true)

      print('Classes for', v, 'output lenght', N)
      save_gold(gold_file, v, probs, indexes, N)

    end
    gold_file:close()
  else
    print("Failed to open", gold_path)
  end
end


--load all images
function load_images(txt_dataset)
  local lines = lines_from(txt_dataset)
  local images = {}
  for k,v in pairs(lines) do
    -- load the image as a RGB float tensor with values 0..1
    local img = image.load(v, 3, 'float')
    local name = v:match( "([^/]+)$" )
    images[k] = {img, name, v}

  end
  return images
end


function test_radiation(arg_size)
  print("Loading gold")
  -- load gold
  local gold_content = load_gold(gold_path)
  if gold_content ~= nil then
    print("Gold loaded with sucess")
  else
    print("Failed load gold")
    return
  end

  -- //////////////////////////////////////////////////////
  -- test
  start_app(weights:match("([^/]+)$"), txt_dataset, gold_path, log)

  -- //////////////////////////////////////////////////////
  local model_obj, softMaxLayer, transform, meanstd = load_model()

  local N = 0
  local images = load_images(txt_dataset)
  for i = 0, iterations do
    for k, v in pairs(images) do

      local img = v[1]
      local name = v[2]
      local raw_name = v[3]

      --    local img = image.load(v, 3, 'float')
      --    local name = v:match( "([^/]+)$" )

      -----------------------------------------------------------------
      local classifing_time = os.clock()
      start_iteration(log)
      -- Scale, normalize, and crop the image
      local img_trans = transform(img)

      -- View as mini-batch of size 1
      local batch = img_trans:view(1, table.unpack(img_trans:size():totable()))

      -- Get the output of the softmax
      local output = model_obj:forward(batch:cuda()):squeeze()

      -- get the dim
      if output:dim() == 1 then
        N = tonumber(#(output:storage()))
      end
      end_iteration(log)
      -- Get the all class indexes and probabilities
      local probs, indexes = output:topk(N, true, true)

      -----------------------------------------------------------------
      print(string.format("Classifing took %.5f seconds", os.clock() - classifing_time))
      local compare_time = os.clock()
      local gold_probs = gold_content[raw_name][1]
      local gold_indexes = gold_content[raw_name][2]


      compare_and_log(log, N, gold_probs, gold_indexes, probs, indexes, raw_name, i)
      print(string.format("Comparing took %.5f seconds", os.clock() - compare_time))

    end
  end

  end_app(log)
end

weights, rad_mode, txt_dataset, gold_path, iterations, log = parse_args()

if rad_mode == nil then

  normal_execution()

elseif rad_mode =="generate" then
  generate_radiation_test(#arg)
elseif rad_mode == "rad_test" then
  test_radiation(#arg)
end
  
  
  
