"""
开发环境: tensorflow 2.0
"""
from utils import load_data, preprocessing_input
from model import TF_verion_2_CNN
import tensorflow as tf
import os
import datetime


def main():
	data_path = "./fer2013/fer2013.csv"
	if not os.path.exists("./model"):
		os.makedirs("./model")
	model_save_path = "./model/TF_verion_2_CNN"+".{epoch:02d}-{loss:.2f}.hdf5"
	# 保存日志
	log_dir= os.path.join("./model","logs_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	# physical_devices = tf.config.experimental.list_physical_devices('GPU')#列出所有可见显卡
	# print("All the available GPUs:\n",physical_devices)
	# if physical_devices:
	# 	gpu=physical_devices[0]#显示第一块显卡
	# 	tf.config.experimental.set_memory_growth(gpu, True)#根据需要自动增长显存
	# 	tf.config.experimental.set_visible_devices(gpu, 'GPU')#只选择第一块


	# 加载人脸表情训练数据和对应表情标签
	faces, emotions = load_data(data_path)
	# 人脸数据归一化，将像素值从0-255映射到0-1之间
	faces = preprocessing_input(faces)
	# 得到表情分类个数
	num_classes = emotions.shape[1]
	# (48, 48, 1)
	image_size = faces.shape[1:]

	batch_size = 64
	num_epochs=50

	#加载模型
	model = TF_verion_2_CNN(image_size, num_classes)

	# tf.keras.utils.plot_model(model,to_file=os.path.join(log_dir,'model.png'),show_shapes=True,show_layer_names=True)


	# 断点续训
	if os.path.exists(model_save_path):
		model.load_weights(model_save_path)
		# 若成功加载前面保存的参数，输出下列信息
		print("checkpoint_loaded")

	# 编译模型，categorical_crossentropy多分类选用
	model.compile(optimizer="SGD",
			  loss="categorical_crossentropy",
			  metrics=["accuracy"])
	# 记录日志
	csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_dir,'logs.log'),separator=',')
	# 保存检查点
	ckpt = tf.keras.callbacks.ModelCheckpoint(
		model_save_path,
		monitor='loss',
		verbose=1,
		save_best_only=False
		)
	model_callbacks = [ckpt, csv_logger]

	# 训练模型
	# https://blog.csdn.net/u011119817/article/details/103181664?ops_request_misc=%25257B%252522request%25255Fid%252522%25253A%252522160864972116780279171504%252522%25252C%252522scm%252522%25253A%25252220140713.130102334.pc%25255Fall.%252522%25257D&request_id=160864972116780279171504&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29_name-16-103181664.pc_search_result_cache&utm_term=tensorflow2.0%E4%B8%ADfit%E5%87%BD%E6%95%B0
	model.fit(x=faces, y=emotions, batch_size=batch_size, epochs=num_epochs,
		verbose=1,
		callbacks=model_callbacks,
		validation_split=0.1)

if __name__ == '__main__':
	main()