require 'nn'
require 'image'
require 'torch'


x1 = 100
y1 = 180
x2 = 1200
y2 = 370
imgCols = y2-y1
imgRows = x2-x1

loaded = torch.load('/home/varun/IITD SRF/code/kitti_data_set_binary/kitti_stereo_dataset_4_cropped.dat')


-- nImages = (#loaded.image_left)[1]

nImagePlanes = 3 -- rgb

-- kitti_stereo_dataset = {
-- image_left = torch.Tensor(nImages,nImagePlanes,imgCols,imgRows),
-- image_right = torch.Tensor(nImages,nImagePlanes,imgCols,imgRows),
-- image_gt_depth = torch.Tensor(nImages,imgCols,imgRows)}

trainData = {
	image_gt_depth = loaded.image_gt_depth:float(),
	image_left = loaded.image_left:float(),
	image_right = loaded.image_right:float()}

print('Converting the images from RGB to YUV')

for i = 1, (#trainData.image_left)[1] do
	trainData.image_left[i] = image.rgb2yuv(trainData.image_left[i])
	trainData.image_right[i] = image.rgb2yuv(trainData.image_right[i]) 
end

print('Normalizing each Y,U,V channel of the left, right images') -- of rgb --> YUV images

channels = {'Y','U','V'}
mean_l = {}
mean_r = {}
std_l = {}
std_r = {}

for i,channel in ipairs(channels) do

	mean_l[i] = trainData.image_left[{ {}, {i}, {}, {} }]:mean()  -- taking the mean and std of just left camera images 
	std_l[i] = trainData.image_left[{{},{i},{},{}}]:std()

	mean_r[i] = trainData.image_right[{{},{i},{},{}}]:mean()    -- taking the mean and std of just right camera images 
	std_r[i] = trainData.image_right[{{},{i},{},{}}]:std()


	trainData.image_left[{{},{i},{},{}}]:add(-mean_l[i])
	trainData.image_left[{{},{i},{},{}}]:div(std_l[i])

	trainData.image_right[{{},{i},{},{}}]:add(-mean_r[i])
	trainData.image_right[{{},{i},{},{}}]:div(std_r[i])
end


print('Normalizing the depth image')

mean_d = trainData.image_gt_depth:mean()
std_d = trainData.image_gt_depth:std()


trainData.image_gt_depth:add(-mean_d)
trainData.image_gt_depth:div(std_d)


print('Normalize all the three channels locally')

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()


--normalize all channels locally
for c in ipairs(channels) do
	for i = 1, (#trainData.image_left)[1] do
		trainData.image_left[{{i},{c},{},{}}] =  normalization:forward(trainData.image_left[{{i},{c},{},{}}])
		trainData.image_right[{{i},{c},{},{}}] =  normalization:forward(trainData.image_right[{{i},{c},{},{}}])
		if c == 1 then
			trainData.image_gt_depth[{{i},{},{}}] = normalization:forward(trainData.image_gt_depth[{{i},{},{}}]) -- normalizing the depth image
		end
	end
end

print('Verify the normalization');

for i,channel in ipairs(channels) do
	trainMean_l = trainData.image_left[{{},{i}}]:mean()
	trainStd_l = trainData.image_left[{{},{i}}]:std()

	trainMean_r = trainData.image_right[{{},{i}}]:mean()
	trainStd_r = trainData.image_right[{{},{i}}]:std()

end

print('train mean left' .. trainMean_l)
print('train mean right' .. trainMean_r)
print('train std left' .. trainStd_l)
print('train std right' .. trainStd_r)




print('Normalizing done.')	


print('Visualize the data now')

-- image.display(trainData.image_left[1])
-- image.display(trainData.image_right[1])
-- image.display(trainData.image_gt_depth[1])



data = {
		mean_left = mean_l,
		mean_right = mean_r,
		std_left = std_l,
		std_right = std_r,
		mean_depth = mean_d,
		std_depth = std_d,
		trainData = trainData}

torch.save('kitti_stereo_dataset_4_normalized_yuv.dat',data)

print('Data object with normalizers saved')

