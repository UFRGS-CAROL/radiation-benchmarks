require 'qt'
require 'torch'
require 'libqttorch'

qt.QImage.fromTensor = function(tensor, scale)
                          return tensor.qttorch.QImageFromTensor(tensor, scale)
                       end

qt.QImage.toTensor = function(self, tensor, scale, depth)
                        if type(tensor) == 'userdata' then
                           return tensor.qttorch.QImageToTensor(self, tensor, scale, depth)
                        else
                           local t = torch.getmetatable(torch.getdefaulttensortype())
                           return t.qttorch.QImageToTensor(self, tensor, scale, depth)
                        end
                     end
