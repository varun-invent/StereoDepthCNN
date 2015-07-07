-- This program gives the basic idea of training a CNN
--Author : Varun Kumar 

require 'torch'
require 'nn'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods


trainData = {data = torch.rand(100,1,41,41),
labels = torch.rand(100,1,41,41)}

--print(labels)
trsize = 100
nstates = {64,64,128} -- hidden neurons for each layer (here 3 hidden layers) conv1,conv2,PreOutput>>>output
filtsize = 7 -- kerner/patch size
poolsize = 2
nfeats = 1
noutputs = 41*41

model = nn.Sequential()

model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))


-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- The feature map size is  64x5x5
-- stage 3: FC NN
model:add(nn.View(nstates[2]*5*5))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[2]*5*5, nstates[3]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[3], noutputs))
model:add(nn.Sigmoid())

--Loss function

criterion = nn.MSECriterion()

-- Some options

opt = {batchSize =10 ,type = 'double'}
--- Train

optimState = {
      learningRate = 1e-3,
      weightDecay = 0,
      momentum = 0,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd  -- using SGD

parameters,gradParameters = model:getParameters()

epoch =  1

-- local vars
local time = sys.clock()

-- set model to training mode (for modules that differ in training and testing, like Dropout)
model:training()

-- shuffle at each epoch
shuffle = torch.randperm(trsize)

-- do one epoch
print('==> doing epoch on training data:')
print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
for t = 1,(#trainData.data)[1],opt.batchSize do
  -- disp progress
  xlua.progress(t, (#trainData.data)[1])

  -- create mini batch
  local inputs = {}
  local targets = {}
  for i = t,math.min(t+opt.batchSize-1,(#trainData.data)[1]) do
     -- load new sample
     local input = trainData.data[shuffle[i]]
     --print('Input size',#input)
     local target = trainData.labels[shuffle[i]]
     --print('traget ',target)
     if opt.type == 'double' then input = input:double()
     elseif opt.type == 'cuda' then input = input:cuda() end
     table.insert(inputs, input)
     table.insert(targets, target)
  end
  --print('Input 1 ',inputs[1])
  --print('Number of inputs ',#inputs)
  --print('no data samples ',(#trainData.data)[1])
  -- create closure to evaluate f(X) and df/dX
  local feval = function(x)
                   -- get new parameters
                   if x ~= parameters then
                      parameters:copy(x)
                   end

                   -- reset gradients
                   gradParameters:zero()

                   -- f is the average of all criterions
                   local f = 0
                   --print('Number of inputs ',#inputs)
                   -- evaluate function for complete mini batch
                   for i = 1,#inputs do
                      -- estimate f
                      local output = model:forward(inputs[i])
                      --print('In loop ',i)
                      --print('output ',output)
                      --print('targets_i ',targets[i])
                      local err = criterion:forward(output, targets[i])
                      f = f + err

                      -- estimate df/dW
                      local df_do = criterion:backward(output, targets[i])
                      model:backward(inputs[i], df_do)

                      -- update confusion
                      --confusion:add(output, targets[i])
                   end

                   -- normalize gradients and f(X)
                   gradParameters:div(#inputs)
                   f = f/#inputs

                   -- return f and df/dX
                   return f,gradParameters
                end

  -- optimize on current mini-batch
  if optimMethod == optim.asgd then
     _,_,average = optimMethod(feval, parameters, optimState)
  else
     optimMethod(feval, parameters, optimState)
  end
end

-- time taken
time = sys.clock() - time
time = time / (#trainData.data)[1]
print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')


