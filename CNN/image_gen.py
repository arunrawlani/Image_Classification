import numpy as np
from PIL import Image

trainX = np.load('tinyX.npy').transpose((0,3,2,1)) # this should have shape (26344, 3, 64, 64)
trainY = np.load('tinyY.npy') 
testX = np.load('tinyX_test.npy').transpose((0,3,2,1)) # (6600, 3, 64, 64)

# to visualize only
# count = 1
# for image_matrix, label in zip(trainX, trainY):
# 	im = Image.fromarray(image_matrix)
# 	im.save("images/"+str(label)+"/image-"+str(count)+".jpg")
# 	print count
# 	count = count + 1
	
count = 1
for image_matrix in testX:
	im = Image.fromarray(image_matrix)
	im.save("test_images/"+"test_img-"+str(count)+".jpg")
	print count
	count = count + 1