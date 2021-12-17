import glob
import cv2
import numpy as np

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.
    
    proc_img = []

    for img in glob.glob("C:/Users/kimbe/Documents/FRI/SB/Regular_track/data/ears/test/*.png"):
        print(img)
        im = cv2.imread(img)

        #sharpening:
        sharpen = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        im = cv2.filter2D(src=im, ddepth=-1, kernel=sharpen)

        #remove noise:
        im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 21)

        im = cv2.addWeighted(im, 0.8, im, 0, 1)

        proc_img.append(im)
        cv2.imwrite("C:/Users/kimbe/Documents/FRI/SB/Regular_track/data/ears/test_preproc/"+  img[-8:], im)
