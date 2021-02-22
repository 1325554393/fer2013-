"""
开发环境: tensorflow 2.0
"""

import tensorflow as tf


#构造模型
#
def TF_verion_2_CNN(input_shape, num_classes):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(32,(1,1), input_shape=input_shape, activation="relu",padding="SAME"))
	model.add(tf.keras.layers.Conv2D(32,(3,3), activation="relu",padding="SAME"))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

	model.add(tf.keras.layers.Conv2D(32,(3,3), activation="relu", padding="SAME"))
	model.add(tf.keras.layers.MaxPooling2D())

	model.add(tf.keras.layers.Conv2D(64,(3,3), activation="relu", padding="SAME"))
	model.add(tf.keras.layers.MaxPooling2D())

	model.add(tf.keras.layers.Flatten())
	# model.add(tf.keras.layers.Dense(512,activation="relu"))
	# model.add(tf.keras.layers.Dropout(0.5))

	model.add(tf.keras.layers.Dense(128,activation="relu"))
	model.add(tf.keras.layers.Dropout(0.5))

	# model.add(tf.keras.layers.Dense(256,activation="relu"))
	# model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(num_classes,activation="softmax"))

	return model

# def main():
# 	input_shape = (48, 48, 1)
# 	num_classes = 7

# 	model = TF_verion_2_CNN(input_shape, num_classes)
# 	model.summary()

# if __name__ == '__main__':
# 	main()


