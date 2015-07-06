--create a module:  nn.Sequence(nn.input -->  nn.Parallel(CONV1-CONV1,CONV2-CONV2)) --> nn.JoinTable() -- >> FC--> Loss


require 'nn'
require 'torch'
require 'image'

-- Load the data---
-- data is in the form of a table which contains three subtables : image_gt_depth,image_left, image_right


loaded = torch.load('/home/varun/IITD SRF/code/kitti_data_set_binary/kitti_stereo_dataset_4_normalized_yuv.dat')

print('loaded',loaded)
trainData = {
	std_right = loaded.std_right,
	std_left = loaded.std_left,
	mean_right = loaded.mean_right,
	mean_left = loaded.mean_left,
	mean_depth = loaded.mean_depth,
	std_depth = loaded.std_depth,
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

-- Patch creation -- 

-- Image size = 190x1100
-- patch Size = 11x11

stride = 41

patch_size = 41

--nPatches =   (size_image_input[2] - patch_size +1) * (size_image_input[3] - patch_size +1) * nImagePairs 

nPatches = math.floor(size_image_input[2]/patch_size)*math.floor(size_image_input[3]/patch_size) * nImagePairs

patch = {
patch_l = torch.Tensor(nPatches, nInputPlanes, patch_size, patch_size),
patch_r = torch.Tensor(nPatches, nInputPlanes, patch_size, patch_size),
patch_d = torch.Tensor(nPatches, patch_size, patch_size)}

patchIndex = 1
for i = 1, nImagePairs do -- For all image pairs
	for j = 1+(patch_size-1)/2 , size_image_input[2]-(patch_size-1)/2, stride do -- For all image patches along the row
		for k = 1+(patch_size-1)/2, size_image_input[3]-(patch_size-1)/2, stride do -- For all image patches along the column
			patch.patch_l[patchIndex] = trainData.image_left[{{i},{},{j-(patch_size-1)/2,j+(patch_size-1)/2},{k-(patch_size-1)/2,k+(patch_size-1)/2}}]
			patch.patch_r[patchIndex] = trainData.image_right[{{i},{},{j-(patch_size-1)/2,j+(patch_size-1)/2},{k-(patch_size-1)/2,k+(patch_size-1)/2}}]
			patch.patch_d[patchIndex] = trainData.image_gt_depth[{{i},{j-(patch_size-1)/2,j+(patch_size-1)/2},{k-(patch_size-1)/2,k+(patch_size-1)/2}}]
			print('patch No '.. patchIndex)
			patchIndex = patchIndex + 1
		end
	end
end

print('Patches created')

print('Patch1 size ',patch.patch_l[1]:size())
image.display(patch.patch_l[1])


------ Convolution -----
--create a module:  nn.Sequence(nn.input -->  nn.Parallel(CONV1-CONV1,CONV2-CONV2)) --> nn.JoinTable() -- >> FC--> Loss

-- patchsize = 11x11
-- stride = 1
-- pooling_stride = poolsize x poolsize
nStates = {64,64,128} -- hidden neurons for each layer (here 3 hidden layers) conv1,conv2,PreOutput>>>output
filtsize = 11 -- kerner/patch size
poolsize = 4
model = nn.Sequential()
for i = 1,2 do
	local parallel = nn.ParallelTable()
	local subModel = nn.Sequential()
	-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
	
	subModel:add(nn.SpatialConvolutionMM(nInputPlanes,nStates[1], filtsize, filtsize))
	subModel:add(nn.ReLU())
    subModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	
	-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    
    subModel:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
    subModel:add(nn.ReLU())
    subModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

    --stage 3: Unrolling into a vector

    subModel:add(nn.view())-----------------------I am here --- Decide what will be the size of final layer
 ------------------------------------------------
 	---Add dropout to subModel
 	---Add Linear to subModel
 	---Add Softmax to subModel
 	---
    -------
    parallel:add(subModel)
end

model:add(parallel)
model:add(nn.JoinTable(1))
--
---Add dropout to Model -----------------------
---Add Linear to Model		This is FC layer
---Add Softmax to Model -----------------------

---Add Loss

-- Remove the spatial normalization of depth map and just normalize wrt constant 255



    