--crop image with x1,y1 = (100,180) to x2,y2 = (1200,370)
-- new image size = 190x1100

require 'nn'
require 'image'
require 'torch'


x1 = 100
y1 = 180
x2 = 1200
y2 = 370
imgCols = x2-x1
imgRows = y2-y1
maxImages = 190 -- Number of image pairs to work on

train_data_path_l = 'training/colored_0/'
train_data_path_r = 'training/colored_1/'
gt_train_path = 'training/disp_noc/'



train_images_list_l = paths.dir(train_data_path_l)
train_images_list_r = paths.dir(train_data_path_r)
gt_train_images_list = paths.dir(gt_train_path)

--print('No of Left images ',#train_images_list_l)
table.sort(train_images_list_l, function (a, b)
      return string.lower(a) < string.lower(b)
    end)

table.sort(train_images_list_r, function (a, b)
      return string.lower(a) < string.lower(b)
    end)

table.sort(gt_train_images_list, function (a, b)
      return string.lower(a) < string.lower(b)
    end)

print('File names Sorted')


--nImages = # gt_train_images_list-2
if maxImages < #gt_train_images_list-2 then
    nImages = maxImages
else
    nImages = #gt_train_images_list-2
end

nImagePlanes = 3 -- rgb
kitti_stereo_dataset = {
image_left = torch.Tensor(nImages,nImagePlanes,imgRows,imgCols),
image_right = torch.Tensor(nImages,nImagePlanes,imgRows,imgCols),
image_gt_depth = torch.Tensor(nImages,1,imgRows,imgCols)}


--train_img_dataset = {}
--gt_train_img_dataset = {}

i=1

for j,img_name in ipairs(train_images_list_l) do
    index,_ = string.find(img_name,'.png')
    if index ~= nill then
        --print(img_name)
        if img_name:sub(index-1,index-1) == '0' then
            print('Image stored' .. i)
            --print(image.crop(image.load(train_data_path_l .. img_name),x1,y1,x2,y2):size())

            --image.display(image.crop(image.load(train_data_path_l .. img_name),x1,y1,x2,y2))
            kitti_stereo_dataset.image_left[i] = image.crop(image.load(train_data_path_l .. img_name),x1,y1,x2,y2)    
        	--print(kitti_stereo_dataset.image_left[1]:size())
        	--image.display(kitti_stereo_dataset.image_left[1])
            kitti_stereo_dataset.image_right[{{i},{},{},{}}] = image.crop(image.load(train_data_path_r .. train_images_list_r[j]),x1,y1,x2,y2)    
            kitti_stereo_dataset.image_gt_depth[{{i},{1},{},{}}] = image.crop(image.load(gt_train_path .. gt_train_images_list[i+2]),x1,y1,x2,y2)
            i = i+1    
        end
    end
    if i == maxImages+1 then
        break
    end
    
end


print('Images stored in table')

--torch.save('kitti_stereo_dataset_100.dat',torch.Tensor(kitti_stereo_dataset))
torch.save('kitti_stereo_dataset_190_cropped.dat',kitti_stereo_dataset)
print('Tensor stored to disc.')
--image.display(train_img_dataset[3])
--image.display(gt_train_img_dataset[3])
