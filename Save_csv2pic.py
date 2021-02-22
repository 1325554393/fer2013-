#import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import os
import cv2

"""
解析csv文件， 并保存为图片
"""


emotions= {
	"0":"anger",
	"1":"disgust",
	"2":"fear",
	"3":"happy",
	"4":"sad",
	"5":"surprised",
	"6":"normal",
}

def createDir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

def load_csv2pic(file):
	#读取csv文件
	faces_data = pd.read_csv(file)
	# print(type(faces_data)) <class 'pandas.core.frame.DataFrame'>
	# print(faces_data)
	image_count = 0
	# loc 定位列表中的行列取值
	for index in range(len(faces_data)):
		emotion_type = faces_data.loc[index][0]
		image_data = faces_data.loc[index][1]
		usage_data = faces_data.loc[index][2]

		# 将图片数据转换成48*48的格式，方便后续图片展示
		# map函数，映射， 将每个值转换为float 类型
		image_data_array = list(map(float, image_data.split()))
		image_data_array = np.asarray(image_data_array)
		# print(image_data_array.shape) 48*48=2304
		image_reshape = image_data_array.reshape(48, 48)
		# print(image_reshape)
		# 
		#选择分类，并创建文件名
		dir_folder = usage_data
		emotion_type_folder = emotions[str(emotion_type)]
		image_reshape_path = os.path.join(dir_folder, emotion_type_folder )

		createDir(dir_folder)
		createDir(image_reshape_path)

		image_reshape_name = os.path.join(image_reshape_path,str(index) + '.jpg')
		#  fer2013都是灰度图，故mode=L,彩色为RGB https://blog.csdn.net/sempronx86/article/details/104096147?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-6.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-6.control
		Image.fromarray(image_reshape.astype('uint8'), mode='L').save(image_reshape_name)
		image_count = index

	print('总共有' + str(image_count) + '张图片')

		# pass
		# print(emotion_data)

def main():
	file = "./fer2013/fer2013.csv"
	load_csv2pic(file)
	pass


if __name__ == '__main__':
	main()

