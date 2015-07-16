-- This code does the following :
-- Read dataset
-- Create patches
-- Creates the learning model  --create a module:  nn.Sequence(nn.input -->  nn.Parallel(CONV1-CONV1,CONV2-CONV2)) --> nn.JoinTable() -- >> FC--> Loss
-- Train the model
-- Author: Varun Kumar


--create a module:  nn.Sequence(nn.input -->  nn.Parallel(CONV1-CONV1,CONV2-CONV2)) --> nn.JoinTable() -- >> FC--> Loss

--require('mobdebug').start()

require 'nn'
require 'torch'
require 'image'
require 'optim'
require 'cutorch'
require 'cunn'

-- Load the data---
-- data is in the form of a table which contains three subtables : image_gt_depth,image_left, image_right


loaded = torch.load('kitti_stereo_dataset_190_yuv_v4.dat')

print('Data loaded',loaded)
trainData = {
	-- std_right = loaded.std_right,
	-- std_left = loaded.std_left,
	-- mean_right = loaded.mean_right,
	-- mean_left = loaded.mean_left,
	--mean_depth = loaded.mean_depth,
	--std_depth = loaded.std_depth,
	image_gt_depth = loaded.trainData.image_gt_depth:float(),
	image_left = loaded.trainData.image_left:float(),
	image_right = loaded.trainData.image_right:float()} 

--print('trainData',trainData)

--print(trainData)
-- Variables defination -- 

 size_image_input = trainData.image_left[1]:size() -- size of One input image

--print('SIze' .. trainData.image_left[1]:size())
nOutputs = size_image_input[2]*size_image_input[3]    -- Will be the depth image

nImagePairs = (#trainData.image_left)[1]

nInputPlanes = size_image_input[1]

-- Date : 09/07/2015-------------
--- Decided to  reduce the input image by half and increasing the number of patches by taking the overlapping patches


-- First lets rescale the images   -- why to rescale the image?? we are working on patches!
-- scaled_image_rows = (#trainData.image_left)[3]/2
-- scaled_image_cols =  (#trainData.image_left)[4]/2

--print('ImageTensor size',#(trainData.image_left[1]))

-- for i = 1,nImagePairs do
--   l = image.scale(trainData.image_left[i],scaled_image_cols,scaled_image_rows);
--   trainData.image_left[i]:resizeAs(l):copy(l)

--   r = image.scale(trainData.image_right[i],scaled_image_cols,scaled_image_rows);
--   trainData.image_right[i]:resizeAs(r):copy(r)

--   d = image.scale(trainData.image_gt_depth[i],scaled_image_cols,scaled_image_rows)
--   trainData.image_gt_depth[i]:resizeAs(d):copy(d)
-- end


-- Patch creation -- 

-- Image size = 190x1100
-- patch Size = 41x41

--stride = 41
stride = 10
patch_size = 41
errors = {}
--nPatches =   (size_image_input[2] - patch_size +1) * (size_image_input[3] - patch_size +1) * nImagePairs 

--nPatches = math.floor(size_image_input[2]/patch_size)*math.floor(size_image_input[3]/patch_size) * nImagePairs

nPatches = math.floor((((#trainData.image_left)[3]-patch_size)/stride)+1)*math.floor((((#trainData.image_left)[4]-patch_size)/stride)+1) * nImagePairs

print('No of patches to be produced ',nPatches)
patch = {
patch_l = torch.Tensor(nPatches, nInputPlanes, patch_size, patch_size),
patch_r = torch.Tensor(nPatches, nInputPlanes, patch_size, patch_size),
patch_d = torch.Tensor(nPatches,1, patch_size, patch_size)}

patchIndex = 1
for i = 1, nImagePairs do -- For all image pairs
	for j = 1+(patch_size-1)/2 , size_image_input[2]-(patch_size-1)/2, stride do -- For all image patches along the row
		for k = 1+(patch_size-1)/2, size_image_input[3]-(patch_size-1)/2, stride do -- For all image patches along the column
			patch.patch_l[patchIndex] = trainData.image_left[{{i},{},{j-(patch_size-1)/2,j+(patch_size-1)/2},{k-(patch_size-1)/2,k+(patch_size-1)/2}}]
			patch.patch_r[patchIndex] = trainData.image_right[{{i},{},{j-(patch_size-1)/2,j+(patch_size-1)/2},{k-(patch_size-1)/2,k+(patch_size-1)/2}}]
			patch.patch_d[patchIndex] = trainData.image_gt_depth[{{i},{1},{j-(patch_size-1)/2,j+(patch_size-1)/2},{k-(patch_size-1)/2,k+(patch_size-1)/2}}]
			print('patch No '.. patchIndex)
			patchIndex = patchIndex + 1
		end
	end
end

print('Patches created')

print('Patch1 size ',patch.patch_l[1]:size())
--image.display(patch.patch_l[1])


------ Convolution ----- Training
--create a module:  nn.Sequence(nn.input -->  nn.Parallel(CONV1-CONV1,CONV2-CONV2)) --> nn.JoinTable() -- >> FC--> Loss

-- Image size = 41x41
-- patchsize = 11x11
-- stride = 1
-- pooling_stride = poolsize x poolsize00 ==

nOutputs = (#patch.patch_d)[3]*(#patch.patch_d)[4]
nstates = {300,200,400} -- hidden neurons for each layer (here 3 hidden layers) conv1,conv2,PreOutput>>>output
filtsize = 7 -- kerner/patch size
poolsize = 2

model = nn.Sequential()
parallel = nn.ParallelTable()

for i = 1,2 do
	
	subModel = nn.Sequential()
	
	-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
	
	subModel:add(nn.SpatialConvolutionMM(nInputPlanes,nstates[1], filtsize, filtsize))
	subModel:add(nn.ReLU())
    subModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	
	-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    
   	subModel:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
   	subModel:add(nn.ReLU())
    subModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

    --stage 3: Unrolling into a vector

	subModel:add(nn.View(nstates[2]*5*5))
	subModel:add(nn.Dropout(0.5))
	subModel:add(nn.Linear(nstates[2]*5*5, nstates[3]))
	subModel:add(nn.ReLU())
	subModel:add(nn.Linear(nstates[3], math.floor(nOutputs/2)))
	subModel:add(nn.ReLU())

    parallel:add(subModel)
    subModel:cuda()
end


model:add(parallel)
model:add(nn.JoinTable(1))   -- Concat two tensors of two images
model:add(nn.Linear(2*math.floor(nOutputs/2), nstates[3]))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[3], nOutputs))
model:add(nn.Tanh())
model:cuda()
parallel:cuda()
---Add Loss
criterion = nn.MSECriterion()
criterion:cuda()


-- print('Calculating sample output')
-- output =  model:forward({patch.patch_l[1],patch.patch_r[1]})   -- Just to test if the model I have made is working fine



print('Starting to train')    

-- Some options

opt = {batchSize =450 ,type = 'cuda'}    
--- Train

optimState = {
      learningRate = 0.3,
      weightDecay = 0.01,
      momentum = 0.01,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd  -- using SGD

parameters,gradParameters = model:getParameters()

--epoch = 1  

-- local vars
  local time = sys.clock()

  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  model:training()


  -- shuffle at each epoch

for epoch = 1,10 do
  
  shuffle = torch.randperm(nPatches)
  --print('# images ',nImagePairs)
  --print('(#patch.patch_l)[1] ',(#patch.patch_l)[1])
  -- do one epoch
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  for t = 1,(#patch.patch_l)[1], opt.batchSize do                                             
    -- disp progress											 
    xlua.progress(t, (#patch.patch_l)[1])							         

    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+opt.batchSize-1,(#patch.patch_l)[1]) do
        --print('Processing in minibatch: image patch no. ',i)
       -- load new sample
       local input_l = patch.patch_l[shuffle[i]]
       local input_r = patch.patch_r[shuffle[i]]
       --print('Input size',#input)
       local target = patch.patch_d[shuffle[i]]:resize(patch.patch_d:size()[3]*patch.patch_d:size()[4])
       --print('traget size',#target)
       if opt.type == 'double' then
       	 input_l = input_l:double()
       	 input_r = input_r:double()
       elseif opt.type == 'cuda' then
  	     input_l = input_l:cuda()
  	     input_r = input_r:cuda()
  	     target = target:cuda()
       end
       table.insert(inputs, {input_l,input_r})
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
                        --output:cuda()
                        --print('In loop ',i)
                        --print('output ',output)
                        --print('targets_i ',targets[i])
                        local err = criterion:forward(output, targets[i])
                        f = f + err

                        -- estimate df/dW
                        local df_do = criterion:backward(output, targets[i])
                        --df_do:cuda()
                        model:backward(inputs[i], df_do)

                        -- update confusion
                        --confusion:add(output, targets[i])
                     end
   -- normalize gradients and f(X)
                     gradParameters:div(#inputs)
                     f = f/#inputs
                     table.insert(errors,f)
                     print('Error in this minibatch is ',f)
                     -- return f and df/dX
                     return f,gradParameters
                  end

    -- optimize on current mini-batch
    
       optimMethod(feval, parameters, optimState)
    
  end
end
-- time taken
time = sys.clock() - time
time = time / (#patch.patch_l)[1]
print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

-- -- Saving the model
-- model_and_normalizers = {
-- model = model,
-- std_right = trainData.std_right,
-- std_left = trainData.std_left,
-- mean_right = trainData.mean_right,
-- mean_left = trainData.mean_left}

torch.save('depth_model_190_samples_v4.dat',model)


print('Model saved')

torch.save('errors.dat',errors)

print('Errors saved')
