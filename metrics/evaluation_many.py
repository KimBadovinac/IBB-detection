import cv2, glob
import numpy as np

class Evaluation:

    def convert2mask(self, mt, shape):
        # Converts coordinates of bounding-boxes into blank matrix with values set where bounding-boxes are.

        t = np.zeros([shape, shape])
        for m in mt:
            x, y, w, h = m
            cv2.rectangle(t, (x,y), (x+w, y+h), 1, -1)
        return t

    def prepare_for_detection(self, prediction, ground_truth):
            # For the detection task, convert Bounding-boxes to masked matrices (0 for background, 1 for the target). If you run segmentation, do not run this function

            if len(prediction) == 0:
                return [], []

            # Large enough size for base mask matrices:
            shape = 2*max(np.max(prediction), np.max(ground_truth)) 
            
            p = self.convert2mask(prediction, shape)
            gt = self.convert2mask(ground_truth, shape)

            return p, gt

    def iou_compute(self, p, gt):
            # Computes Intersection Over Union (IOU)
            if len(p) == 0:
                return 0

            intersection = np.logical_and(p, gt)
            union = np.logical_or(p, gt)

            iou = np.sum(intersection) / np.sum(union)

            return iou

if __name__ == '__main__':
    count = 0
    eval_sum = 0
    for fname1 in glob.glob("C:/Users/kimbe/Documents/FRI/SB/Regular_track/data/ears/annotations/segmentation/test/*.png"):
        #fname2 = "C:/Users/kimbe/Documents/FRI/SB/Regular_track/data/ears/test/detected_cascade_eval/" + fname1[-8:-3] + "png.eval.jpg"
        fname2 = "C:/Users/kimbe/Documents/FRI/SB/Regular_track/data/ears/test/detected_yolo_eval/" + fname1[-8:-3] + "png.eval.jpg"

        img1 = cv2.imread(fname1)
        img2 = cv2.imread(fname2)
        img1 = cv2.resize(img1, None, fx=0.4, fy=0.4) ##
        
        height, width = img1.shape[:2]
        eval = Evaluation()

        #cv2.imshow('a',img1)
        #cv2.waitKey()
        #cv2.imshow('b', img2)
        #cv2.waitKey()
        
        eval_sum += eval.iou_compute(img1,img2)
        count+=1
    print(eval_sum/count)

