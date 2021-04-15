# This is only for testing
from pytorch import *
from drawing import *
import cv2 as cv
from skimage.color import rgb2gray

model = torch.load('my_model_lin.pth')

img = cv.imread('digit_inv_28x28.jpg')  # 28 x 28 x 3. Here 255 is white and 0 is black. cv reads it in as an np.array!!
# no need to invert (already done) and no need to resize, but do need to make grayscale again.
img = rgb2gray(img) # 28 x 28. makes grayscale - automatically makes it between 0-1 too.
tensor = torch.tensor(img)  # 2d tensor 28 x 28.
tensor = tensor.flatten() # 1d tensor 784 length.

yhat = predict(tensor,model)
print(yhat)

# plt.imshow(img,cmap='gray')  
# plt.show()
# print(img)
# print(img.shape)
# print(img.size)
# print(img.dtype)
