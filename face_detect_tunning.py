import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


"""
haar 检测器，本质上是色差对比，例如：眼睛的周边要比眼睛本身亮， 鼻梁的本身要比周边亮
然后创建haar检测器，白色和黑色的检测器，分为水平，垂直，倾斜
scaler就是检测器的大小，沿着图片去寻找这些特征点，scaler越小，检测的次数越多
，检测过程中，白色特征总和减去黑色特征，差值过大的，符合要求的记为特征点
利用这些特征点，检测到人脸

因此，若图片的亮度过大，色差不明显， 人脸占比较大，都不容易检测出来

需通过调参，进行适度的识别精度提高
实验过程中，alt检测效果要好于default，但两者对平常自己拍的照片检测精度都过低

对于大型项目，应训练自己的人脸识别模型


"""

detection_model_path = './model/face_detect/haarcascade_frontalface_alt.xml'
pic_path = "./pic_test/5.jpg"
face_detection = cv2.CascadeClassifier(detection_model_path)
pil_image = image.load_img(pic_path, grayscale=True,color_mode = "grayscale")
gray_image = image.img_to_array(pil_image)

gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype("uint8")

def main():
	img = cv2.imread(pic_path)
	#print(type(img))
	#print(img)
	# 1.02： 可以理解为检测次数，比较细腻， 一般为（1.0~1.5） 越小表示检测越仔细，检测的可能人脸数越多，
	# 1：为重复次数，在检测很多次数的情况下，每次都出现的个数，1,是个阈值，低于的排除
	faces = face_detection.detectMultiScale(gray_image, 1.02, 1)
	print(faces)
	for face_coordinates in faces:
		x, y, width, height = face_coordinates
		cv2.rectangle(img, (x,y),(x+width, y+height),(0,0,255), 2)
		# image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		#cv2.imshow(image1)
		cv2.imwrite("./pic_test/"+"predict"+os.path.basename(pic_path), img)
	print("done")

if __name__ == '__main__':
	main()
