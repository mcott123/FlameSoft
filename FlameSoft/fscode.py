# coding: utf-8

# In[4]:


import cv2
import numpy as np
from pynput.keyboard import Key, Controller

keyboard = Controller()


class fs(object):

    def __init__(self, path):
        self.points = []
        self.crop = False
        self.path = path

    def area(self):
        """Crop the image"""

        def mouse(event, x, y, flags, param):
            global refpt, crop

            if event == cv2.EVENT_LBUTTONDOWN:
                crop = True
                self.points.append((x, y))


            elif event == cv2.EVENT_LBUTTONUP:
                crop = False
                self.points.append((x, y))

                """Alows to press and release keys"""
                keyboard.press(Key.esc)
                keyboard.release(Key.esc)

        cap = cv2.VideoCapture(0)

        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", mouse)

        """Extracting Area of Interest from Image"""

        img = cv2.imread(self.path)
        red = img[:, :, 2]
        cv2.imshow('frame', red)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        dx = self.points[1][0] - self.points[0][0]
        dy = self.points[1][1] - self.points[0][1]
        return self.points

    def cropVideo(self):

        """Crop the image"""

        def mouse(event, x, y, flags, param):
            global refpt, crop

            if event == cv2.EVENT_LBUTTONDOWN:
                crop = True
                self.points.append((x, y))


            elif event == cv2.EVENT_LBUTTONUP:
                crop = False
                self.points.append((x, y))

                """Alows to press and release keys"""
                keyboard.press(Key.esc)
                keyboard.release(Key.esc)

        cap = cv2.VideoCapture(self.path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = (cap.get(cv2.CAP_PROP_FPS))
        step = int(length / fps) / 3
        cap.set(cv2.CAP_PROP_POS_MSEC, step * 1000)

        success, image = cap.read()

        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", mouse)

        """Extracting Area of Interest from Image"""
        red = image[:, :, 2]
        cv2.imshow('frame', red)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return self.points


# In[5]:


class vid(object):
    def __init__(self, path, points, filtersize, pixelBrightness):
        self.path = path
        self.points = points
        self.pixelBrightness = pixelBrightness
        self.x = []
        self.y = []
        self.filter = filtersize
        self.edgemat = []
        self.insFPS = 0

    def edge(self):
        cap = cv2.VideoCapture(self.path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.insFPS = (cap.get(cv2.CAP_PROP_FPS))
        duration = length / self.insFPS
        dx = self.points[1][0] - self.points[0][0]  # pixels in x direction
        dy = self.points[1][1] - self.points[0][1]  # pixels in y direction

        ##          """ Mask for the Back Ground Substractor"""
        # fgbg = cv2.createBackgroundSubtractorMOG2()

        while True:
            # Reading Frame
            ret, frame = cap.read()

            if ret == False:
                break

            # Extracting the clicked frame from mouse events
            red = frame[:, :, 2]

            img = red[self.points[0][1]:self.points[1][1], self.points[0][0]:self.points[1][0]]

            # Apply Gaussian Filter
            img_blur = cv2.blur(img, (self.filter, self.filter))

            # Apply image threshholding
            ret1, thresh = cv2.threshold(img_blur, self.pixelBrightness, 255, cv2.THRESH_BINARY)

            # Apply the Background Substractor
            #     fgmask = fgbg.apply(img, learningRate=0.001)      #Uncheck to activate

            # Apply Canny Edge Detection
            edge = cv2.Canny(thresh, 175, 200)  # Edge from threshholding

            #     edge=cv2.Canny(fgmask,175,255) #Edge from Backgroun Substractor Uncheck to activate

            ###        """ Storing the data for the  calculation"""

            self.edgemat.append(edge)

            concat = np.vstack((img, thresh, edge))
            cv2.imshow('Flame', concat)

            k = cv2.waitKey(10) & 0xff
            # Press Q on keyboard to  exit
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        return self.edgemat


# In[7]:


if __name__ == "__main__":
    crop = fs('A01.avi')
    a = fs('A01.avi').cropVideo()
# #     crop=[(91, 356), (969, 570)]
#     aa=vid('A01.avi',a,75,150).edge()
