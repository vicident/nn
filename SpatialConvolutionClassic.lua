local THNN = require 'nn.THNN'
local SpatialConvolutionClassic, parent = torch.class('nn.SpatialConvolutionClassic', 'nn.Module')
function SpatialConvolutionClassic:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   
   self:reset()
end

function SpatialConvolutionClassic:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end) 
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function SpatialConvolutionClassic:updateOutput(input)
      input.THNN.SpatialConvolution_updateOutput(
        input:cdata(),
        self.output:cdata(),
        self.weight:cdata(),
        THNN.optionalTensor(self.bias),
        self.dW, self.dH)
   return self.output
end

function SpatialConvolutionClassic:updateGradInput(input, gradOutput)
   return nil
end

function SpatialConvolutionClassic:accGradParameters(input, gradOutput, scale)
   return nil
end

function SpatialConvolutionClassic:type(type,tensorCache)
   return parent.type(self,type,tensorCache)
end

function SpatialConvolutionClassic:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
    if self.bias then
      return s .. ')'
   else
      return s .. ') without bias'
   end
end

function SpatialConvolutionClassic:clearState()
   return parent.clearState(self)
end
