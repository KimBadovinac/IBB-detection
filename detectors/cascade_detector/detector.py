import cv2, sys, os

class CascadeDetector:
	# This example of a detector detects faces. However, you have annotations for ears!
	#cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'lbpcascade_frontalface.xml'))

  #use pretrained xml file for cascade classifier
	cascade_left_ear = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
	cascade_right_ear = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

	def detect(self, img):
		left_ear = self.cascade_left_ear.detectMultiScale(img, 1.05, 1)
		right_ear = self.cascade_right_ear.detectMultiScale(img, 1.05, 5)
		return left_ear, right_ear

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	detector = CascadeDetector()
	detected_left, detected_right = detector.detect(img)
	
	for x, y, w, h in detected_left:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 3)
		print(x, y, w, h)
	for x, y, w, h in detected_right:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 3)
		print(x, y, w, h)
	
	cv2.imwrite(fname + '.detected.jpg', img)
	cv2.imshow('Ear Detector', img)
	cv2.waitKey()
	cv2.destroyAllWindows()
