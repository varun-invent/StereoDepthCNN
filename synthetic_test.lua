require 'torch'
require 'image'
require 'nn'
require 'optim'

loaded = torch.load('test.dat')
loaded1 =  torch.load('depth_model_190_samplesv2.dat')

model = loaded1.model

image_l = loaded[1]
image_r = loaded[2]

image_l = image.rgb2yuv(image_l)
image_r =  image.rgb2yuv(image_r) 


patch_size = 41
stride_testing =  10 	

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


