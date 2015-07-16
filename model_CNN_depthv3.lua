-- This code tests the model prepared in model_CNN_depthv2.lua
-- By reading the images
-- converting it to tensors
-- normalizing it
-- Breaking it into patches

--Author : Varun Kumar

require 'torch'
require 'image'
require 'nn'
require 'optim'


loaded =  torch.load('depth_model_190_samplesv2.dat')

model = loaded.model
mean_left = loaded.mean_left
mean_right =  loaded.mean_right
std_left = loaded.std_left
std_right = loaded.std_right

model:evaluate()
-- params for cropping the image
x1 = 100
y1 = 180
x2 = 1200
y2 = 370
imgCols = y2-y1
imgRows = x2-x1

-- Test Image path

test_data_path_l = 'training/colored_0/'    ------------Taking the same data set used in training------------- 
test_data_path_r = 'training/colored_1/'
gt_test_path = 'training/disp_noc/'

test_images_list_l = paths.dir(test_data_path_l)
test_images_list_r = paths.dir(test_data_path_r)
gt_test_images_list = paths.dir(gt_test_path)

-- sorting the image names

table.sort(test_images_list_l, function (a, b)
      return string.lower(a) < string.lower(b)
    end)

table.sort(test_images_list_r, function (a, b)
      return string.lower(a) < string.lower(b)
    end)

table.sort(gt_test_images_list, function (a, b)
      return string.lower(a) < string.lower(b)
    end)

--print(test_images_list_l)
print('File names Sorted')

-- Read the images
--image_left = torch.Tensor(1,1,)

imageIndex = 5    -- Index of which image to use . It starts with 3 as in linux file system the 1st is '.' and 2md is '..'

image_l = image.crop(image.load(test_data_path_l .. test_images_list_l[imageIndex]),x1,y1,x2,y2):float()     -------------Interesting fact is that the spatial normalization 
image_r = image.crop(image.load(test_data_path_r .. test_images_list_r[imageIndex]),x1,y1,x2,y2):float()     -----needs float input, but the model:forward() needs double input types 
image_d = image.crop(image.load(gt_test_path .. gt_test_images_list[4]),x1,y1,x2,y2):float()


--print('image_l',image_l)

--print('image_left',#image_left)

-- Convert RGB to YUV images

image_l = image.rgb2yuv(image_l)
image_r =  image.rgb2yuv(image_r) 



-- Swap the spatial normalization and layerwise normalization   - Date 09/07/2015 ----

-- channels = {'y','u','v'}

-- -- Now conver the images into 
-- -- Spatial normalize each channel layer------------
-- -- For spatially normalizing the images------------
--  -- Define the normalization neighborhood:
-- neighborhood = image.gaussian1D(13)

-- -- Define our local normalization operator (It is an actual nn module, 
-- -- which could be inserted into a trainable model):
-- normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- --print('# image', #image_l[{{1},{},{}}])

-- for i,channel in ipairs(channels) do

-- 	image_l[i] = normalization:forward(image_l[{{i},{},{}}])
-- 	image_r[i] = normalization:forward(image_r[{{i},{},{}}])



-- --Normalize the L,R images channel wise along the channels

-- for i,channel in ipairs(channels) do

	
-- 	image_l[i] = image_l[i]:add(-mean_left[i]):div(std_left[i])
-- 	image_r[i] = image_r[i]:add(-mean_right[i]):div(std_right[i])



-- end


-- end

-- print('Normalized')
-- ----------------------------


-- Rescale the images ---------------------Date 09/07/2015 -----
-- scaled_image_rows = (#image_l)[2]/2
-- scaled_image_cols = (#image_l)[3]/2

-- l = image.scale(image_l,scaled_image_cols,scaled_image_rows);
-- image_l:resizeAs(l):copy(l)

-- r = image.scale(image_r,scaled_image_cols,scaled_image_rows);
-- image_r:resizeAs(r):copy(r)

-- d = image.scale(image_d,scaled_image_cols,scaled_image_rows);
-- image_d:resizeAs(d):copy(d)






--test output
--print('Computing OUT')
--print('Size patch :',image_l[{ {},{1,41},{1,41} }])
--out = model:forward({image_l[{ {},{1,41},{1,41} }],image_r[{ {},{1,41},{1,41} }]})
--print('OUT computed')
-- Test on patches-----------
patch_size = 41
stride_testing = 5  	

nPatches = math.floor((((#image_l)[2]-patch_size)/stride_testing)+1)*math.floor((((#image_l)[3]-patch_size)/stride_testing)+1)

outputDepth = torch.Tensor(1,(#image_l)[2],(#image_l)[3])
for i = 1, (#image_l)[2], stride_testing do
	
	for j = 1,(#image_l)[3], stride_testing do
		print('i and j',i,j)
		--print('Tensor test',image_l[{{1},{i,i+patch_size-1},{j,j+patch_size-1}}])

		out = model:forward({image_l[{{},{i,i+patch_size-1},{j,j+patch_size-1}}]:double(),image_r[{{},{i,i+patch_size-1},{j,j+patch_size-1}}]:double()})	

		outputDepth[{{1},{i,i+patch_size-1},{j,j+patch_size-1}}] =  out:resize(1,patch_size,patch_size)

		if j+stride_testing+patch_size > (#image_l)[3] then
			break
		end

	end
	
		if i+stride_testing+patch_size > (#image_l)[2] then
			break
		end

end

image.display(outputDepth)

