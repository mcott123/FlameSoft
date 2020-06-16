# coding: utf-8

# In[4]:

import cv2 as cv
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
        point = [(73, 93), (990, 711)]
        success, image = cap.read()
        # image = image[point[0][1]:point[1][1], point[0][0]:point[1][0]]
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

        # Get the first frame
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, image0 = cap.read()
        red1 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)

        ##          """ Mask for the Back Ground Substractor"""
        # fgbg = cv2.createBackgroundSubtractorMOG2()

        while True:
            # Reading Frame
            ret, frame = cap.read()

            if ret == False:
                break

            # Extracting the clicked frame from mouse events
            red = frame[:, :, 2] #- red1

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

            ## Trials
            img = cv2.cvtColor(frame, cv.COLOR_BGR2GRAY)[self.points[0][1]:self.points[1][1], self.points[0][0]:self.points[1][0]]

            img = img - red1[self.points[0][1]:self.points[1][1], self.points[0][0]:self.points[1][0]]

            # img = cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
            # img = cv2.Laplacian(thresh, cv2.CV_64F)



            # concat = np.vstack((img, thresh, edge))
            cv2.imshow('Flame', img)

            k = cv2.waitKey(10) & 0xff
            # Press Q on keyboard to  exit
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        return self.edgemat

class vid_check(object):
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
        print(length, self.insFPS)
        duration = length / self.insFPS
        dx = self.points[1][0] - self.points[0][0]  # pixels in x direction
        dy = self.points[1][1] - self.points[0][1]  # pixels in y direction

        # Get the first frame
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, image0 = cap.read()
        red1 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)

        c = self.points[1][0]-self.points[0][0]
        ans = np.empty(c,)
        break_point = 4
        x_pixel = self.points[1][0] - self.points[0][0]
        breaks = np.append(np.arange(0, x_pixel, x_pixel/break_point).astype(int), x_pixel)


        while True:
            # Reading Frame
            ret, frame = cap.read()


            if ret == False:
                break

            # Extracting the clicked frame from mouse events
            img = frame[:, :, 2] #- red1

            ## Trials
            img = cv2.cvtColor(frame, cv.COLOR_BGR2GRAY)[self.points[0][1]:self.points[1][1], self.points[0][0]:self.points[1][0]]
            img2 = img

            # img = img - red1[self.points[0][1]:self.points[1][1], self.points[0][0]:self.points[1][0]]
            img_blur = cv2.blur(img, (self.filter, self.filter))
            # Apply image threshholding
            ret1, thresh = cv2.threshold(img_blur, self.pixelBrightness, 255, cv2.THRESH_BINARY)

            # img = cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
            img = cv2.Laplacian(thresh, cv2.CV_64F)

            # print(val.shape)
            # print(img[230, :].shape)
            # print(img.shape)
            ans = np.vstack((ans, thresh[560, :]))
            cc = np.vstack((thresh, img2))
            # concat = np.vstack((img_blur, ans))

            # concat = np.vstack((img, thresh, edge))
            cv2.imshow('Flame', ans)
            cv2.imshow('blur', cc)

            k = cv2.waitKey(10) & 0xff
            # Press Q on keyboard to  exit
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        return ans

# In[7]:


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    path = r'E:\Github\Flame-Speed-Tool\bin\test.avi'
    # crop = fs(path).cropVideo()
    # # print(crop)
    point = [(66, 0), (963, 575)]
    aa = vid_check(path, points=point, filtersize=175, pixelBrightness=150).edge()
    #
    # aa = cv2.imread(r'E:\Github\Flame-Speed-Tool\bin\00307.jpg')[:, :, 2][point[0][1]:point[1][1], point[0][0]:point[1][0]]
    # # x = i[230, :]
    # # y = i[210, :]
    # v = np.median(aa)
    # lower = int(max(0, (1.0 - 0.33) * v))
    # upper = int(min(255, (1.0 + 0.33) * v))
    # aa = aa.astype(np.uint8)
    # aa = cv2.Canny(aa, lower, upper)  # Edge from threshholding
    # cv2.imshow('frame', aa)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # x, y = aa.shape
    # x_pos = np.arange(0, x, 100)
    # # y_pos = np.arange(y, 0, -1)
    # # plt.xticks(x_pos, x_pos)
    # plt.imshow(aa.transpose())
    # plt.xticks(x_pos, x_pos)
    # plt.show()


    # cap = cv2.VideoCapture(path)
    # cap.set(cv2.CAP_PROP_POS_MSEC, 12000)
    # success, image0 = cap.read()
    #
    # points = [(97, 360), (966, 575)]
    # red1 = image0[:, :, 2][points[0][1]:points[1][1], points[0][0]:points[1][0]]
    # red1 = cv2.applyColorMap(red1, cv2.COLORMAP_JET)
    #
    # red1 = cv2.Laplacian(red1, cv2.CV_64F)
    #
    #
    # # x = cv2.calcHist(red1, [0], None,[256],[0,256])
    # # plt.plot(x)
    # # plt.show()
    #
    #
    # cv2.imshow('frame', red1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    pass