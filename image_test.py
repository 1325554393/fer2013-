import cv2
import os
import tensorflow as tf
import numpy as np
from utils import preprocessing_input, load_image, get_coordinates, detect_faces, draw_bounding_box, draw_text

# extensions = [".jpg", "png", "jpeg"]

text_floder = "./pic_test"
dirs = os.listdir(text_floder)

emotion_labels = {0: 'angry', 1: 'disgust', 2: 'sad', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}


# 加载cv2中的人脸识别模型
detection_model_path = './model/face_detect/haarcascade_frontalface_alt.xml'
emotion_model_path = './model/TF_verion_2_CNN.22-1.47.hdf5'
# 加载人脸识别模型，
face_detection = cv2.CascadeClassifier(detection_model_path)
# 加载表情识别模型
emotion_classifier = tf.keras.models.load_model(emotion_model_path, compile=False)
# print(emotion_classifier) <tensorflow.python.keras.engine.sequential.Sequential object at 0x000000000D5C1988>
# 获得模型输入图形宽高尺寸大小 48*48
emotion_target_size = emotion_classifier.input_shape[1:3]
#print(emotion_target_size)

def main():
	#开始测试
	for file in dirs:
		image_path = os.path.join(text_floder, os.path.basename(file))
		#print(os.path.basename(file))
		#print(image_path)
		# 加载灰度图像和灰度图片
		rgb_image = load_image(image_path, grayscale=False, color_mode = "rgb")
		gray_image = load_image(image_path, grayscale=True,color_mode = "grayscale")
		# 去掉维度为1的维度（只留下宽高，去掉灰度维度）
		gray_image = np.squeeze(gray_image)
		gray_image = gray_image.astype("uint8")
		faces = detect_faces(face_detection, gray_image)
		#print("-----")
		#print(len(faces))
		for face_coordinates in faces:
			#获取人脸在图像中的矩形坐标的对角两点
			x1, x2, y1, y2 = get_coordinates(face_coordinates)
			#print(x1, x2, y1, y2 )
			# 截取人脸图像像素数组
			gray_face = gray_image[y1:y2, x1:x2]

			try:
				# 将人脸reshape模型需要的尺寸
				gray_face = cv2.resize(gray_face,(emotion_target_size))
			except:
				continue

			gray_face = preprocessing_input(gray_face)
			gray_face = np.expand_dims(gray_face, 0)
			gray_face = np.expand_dims(gray_face, -1)
			# print(gray_face.shape)

			# 预测
			emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
			emotion_text = emotion_labels[emotion_label_arg]

			color = (255,0,0)
			# 画边框
			draw_bounding_box(face_coordinates, rgb_image, color)
			# 画表情说明
			draw_text(face_coordinates, rgb_image, emotion_text, color, 0, face_coordinates[3]+30, 1, 2)

			# 将图像转换为BGR模式显示
			bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
			cv2.imwrite("./pic_test/"+"predict"+os.path.basename(file), bgr_image)

			cv2.waitKey(1)
			cv2.destroyAllWindows()

	print("已识别%d张图片" % int(len(dirs)))


if __name__ == '__main__':
	main()


# image_path = "./pic_test/1.jpg"

# gray_image = load_image(image_path, grayscale=True,color_mode = "grayscale")
# gray_image = np.squeeze(gray_image)
# gray_image = gray_image.astype("uint8")
# faces = detect_faces(face_detection, gray_image)
# print(faces)
# draw_bounding_box(face_coordinates, rgb_image, color)
# bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
# cv2.imwrite("./pic_test/"+"predict"+os.path.basename(file), bgr_image)





