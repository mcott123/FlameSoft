import os
from sys import path

path.append(r'E:\Github\Flame-Speed-Tool\FlameSoft')
import cv2 as cv2
import numpy as np
from pandas import read_csv


class Crop(object):

    def __init__(self, out: str = None, path: str = None):
        """The crop class to get the pixel values for the cropped image
        path: string array to the path
        """
        self.path = path
        self.points = [(), ()]
        self.image = None
        self.out = out

    def mouse_crop(self, event, x, y, flags, param):
        """Method to get the x and y locations on the image"""

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[0] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.points[1] = (x, y)

            # Draw the rectangle around the slected path
            cv2.rectangle(self.image, self.points[0], self.points[1], (255, 0, 0), 3)
            cv2.imshow('Frame', self.image)

    def crop_video(self):
        """Method to get the cropped points location from the video"""

        cap = cv2.VideoCapture(self.path)

        # Set the frame 1 as image from which it will be cropped
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)

        # Read the imaage at ste it as class attribute
        success, self.image = cap.read()

        # Name the window and usual code to view image
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', self.mouse_crop)
        cv2.imshow('Frame', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        with open(self.out + r'\bin' + r'\points.txt', 'w+') as out:
            for tp in self.points:
                for val in tp:
                    out.write(str(val) + '\n')

        return self.points

    def crop_image(self, path):
        image = cv2.imread(path)

        self.image = image
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', self.mouse_crop)
        cv2.imshow('Frame', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.points


class Flame(object):
    textpath = ""
    edgepath = ""
    imagepath = ""
    arraypath = ""
    outpath = ""

    def __init__(self, path: str, out: str):
        self.path = path
        self.out = out
        if not os.path.exists(self.out + r'\bin'):
            os.mkdir(self.out + r'\bin')
        Flame.textpath = self.out + r'\bin' + r'\text.txt'
        Flame.edgepath = self.out + r'\bin' + r'\edge.png'
        Flame.imagepath = self.out + r'\bin' + r'\process.png'
        Flame.arraypath = self.out + r'\bin' + r'\numpy'
        Flame.outpath = self.out + r'\bin' + r'\output.xlsx'

    def process(self, breaks: int, filter_size: list, thresh_val: list, crop_points: list, flow_right: bool):

        # Assert check for the length of inputs
        if len(filter_size) != breaks or len(thresh_val) != breaks:
            raise AssertionError("Length of Filter Size and Thresh Val == Breaks")

        # Capture the video
        cap = cv2.VideoCapture(self.path)
        success, frame = cap.read()
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[crop_points[0][1]:crop_points[1][1],
                 crop_points[0][0]:crop_points[1][0]]
        # Break the image inot parts
        length, width = frame1.shape
        array = self.break_image(breaks, (length, width))

        # Get video properties and make an empty numpy array to store the results(array shape frames * length)
        cap_fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        cap_duration = cap_fcount / cap_fps
        ans = np.zeros((cap_fcount, crop_points[1][0] - crop_points[0][0]))

        # Make the dictionary of the functions to store the broken images
        fname = {}
        blur = {}
        view = {}
        thresh = {}
        for index, val in enumerate(array):
            fname[f'frame{index}'] = None
            blur[f'blur{index}'] = None
            view[f'view{index}'] = None
            thresh[f'thresh{index}'] = None

        # Start counting the frames ( appending ans matrix)
        frame_count = 0
        while True:

            # Read the frames
            success, frame = cap.read()

            # Break the loop on the last frame
            if not success:
                break

            # Convert the frame  to grayscale and crop the frame as per points
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[crop_points[0][1]:crop_points[1][1],
                    crop_points[0][0]:crop_points[1][0]]

            # Substract the first frame from subsequent frames to reduce noise
            frame = frame - frame1

            # Initiate count to count all the broken images
            count = 0
            for key, val, blur_key, view_key, thresh_key in \
                    zip(fname.keys(), array, blur.keys(), view.keys(), thresh.keys()):

                # Divide the Frame by iterating over the array
                fname[key] = frame[:, val[0]:val[1]]

                # Blur the image
                blur[blur_key] = cv2.blur(fname[key], (filter_size[count], filter_size[count]))

                # Thresh the image
                _, thresh[thresh_key] = cv2.threshold(blur[blur_key], thresh_val[count], 255, cv2.THRESH_BINARY)
                # thresh[thresh_key] = cv2.blur(thresh[thresh_key], (25, 25))
                # _, thresh[thresh_key] = cv2.threshold(thresh[thresh_key], 100, 255, cv2.THRESH_BINARY)

                # Add the images vertically together
                view[view_key] = cv2.vconcat([fname[key], blur[blur_key], thresh[thresh_key]])

                # Generate the string code to be execeuted to stich the images together horizontally
                code2 = "cv2.hconcat(["
                count2 = 0
                for key, val1 in zip(view.keys(), view.values()):
                    if val1 is not None:
                        if count2 < 1:
                            code2 = code2 + f"view['{key}']"

                        else:
                            code2 = code2 + "," + f"view['{key}']"
                        count2 = count2 + 1

                code2 = code2 + "])"

                count = count + 1

            # Execute the code2 generates above and assign the value to variable views
            views = eval(code2)

            # Get the slice of the pixels along the depth of image
            slice_val = fname['frame0'].shape[0] * 2 + int(fname['frame0'].shape[0] / 1.1)

            # Adjust the array for the flame flow
            if flow_right:
                ans[frame_count, :] = views[slice_val, :]
            else:
                ans[frame_count, :] = views[slice_val, :][::-1]

            # Increase the frame count
            frame_count = frame_count + 1
            # self.pimage = ans
            # Resize the images
            ans_img = cv2.resize(ans, (1080, 360))
            views = cv2.resize(views, (1020, 780))
            # Show the images
            cv2.imshow('Thresh', ans_img)
            cv2.imshow('view', views)
            k = cv2.waitKey(10) & 0xff
            # Press esc on keyboard to  exit
            if k == 27:
                break
        # Save the image to numpy arrray
        np.save(Flame.arraypath, ans)
        cv2.imwrite(Flame.imagepath, ans)
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def break_image(num: int, shape: tuple):
        """Method to break the image """
        # Make a array based on number and shape of tuple(shape of image)
        array = np.append(np.arange(0, shape[1], int(shape[1] / num)), shape[1])
        # Create the empty list to be appended
        ans = []
        # Iterate to get the tuples of slices
        for index, val in enumerate(array):
            # Append while index < len(array) - 1
            if index < len(array) - 1:
                ans.append((int(array[index]), int(array[index + 1])))

        # if the differecnce in last break is less than 20 the delete and replace that with prior tuple
        if ans[-1][1] - ans[-1][0] < 10:
            val1 = ans[-1][1]
            val0 = ans[-2][0]
            ans.pop(-1)
            ans.pop(-1)
            ans.append((val0, val1))

        return ans

    def whiten_image(self, path_: str = None):
        """Method to whited the pixels of the image before edge detection"""
        # Check if the path is provided else use class variable
        try:
            if path_ is None:
                path_ = Flame.imagepath
            # Get the points to be blackened
            points = Crop().crop_image(path_)
            img = cv2.imread(path_)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Get the values of the points sorted for the slice of the array
            x_start = min(points[0][0], points[1][0])
            if x_start < 0:
                x_start = 0
            x_end = max(points[0][0], points[1][0])
            y_start = min(points[0][1], points[1][1])
            if y_start < 0:
                y_start = 0
            y_end = max(points[0][1], points[1][1])

            # Assign pixels the value of 255 (white)
            img[y_start:y_end, x_start:x_end] = 255

            # Write the image to path
            cv2.imwrite(path_, img)

        except Exception as e:
            print(e)

        return 0

    def blacken_image(self, path_: str = None):
        """Method to blacken the pixels of image before edge detection"""

        # Check if the path is provided else use class variable
        if path_ is None:
            path_ = Flame.imagepath

        # Get the points to be blackened
        points = Crop().crop_image(path_)
        img = cv2.imread(path_)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get the values of the points sorted for the slice of the array
        x_start = min(points[0][0], points[1][0])
        if x_start < 0:
            x_start = 0
        x_end = max(points[0][0], points[1][0])
        y_start = min(points[0][1], points[1][1])
        if y_start < 0:
            y_start = 0
        y_end = max(points[0][1], points[1][1])

        # Assign pixels the value of 0 (black)
        img[y_start:y_end, x_start:x_end] = 0

        # Write the image to path
        cv2.imwrite(path_, img)

        return 0

    def get_data(self):
        """Method to get the edge from the threshed imaage and save data to test and npy array"""

        # Read the image
        img = cv2.imread(self.imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Run Canny edge detector with default arguments
        edge = cv2.Canny(img, 100, 200)

        # Get the x and y arrays for the edge using np.where
        x, y = np.where(edge == 255)

        # Reshape and  array so that it could be written ( row array to column array)
        ans = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

        # Write the image to the path
        cv2.imwrite(Flame.edgepath, edge)
        np.savetxt(Flame.textpath, ans, delimiter=',')

        return ans

    @staticmethod
    def Dataframe(length: float, fps: float):
        """Method to process the text data to the Dataframe and save to output.xlsx"""

        # Make the dataframe from the text path
        df = read_csv(Flame.textpath, sep=',', header=None)
        df.columns = ['Frame', 'XPixel']

        # Read the image to get the value for the y pixels
        img = cv2.imread(Flame.edgepath)
        pixels = img.shape[1]

        # Pixel length = length / number of pixels
        pixel_length = round(length / pixels, 6)

        # Make the columns for time and distance
        df['Distance (ft)'] = df['XPixel'] * pixel_length
        df['Time (msec)'] = df['Frame'] * 1000 / fps

        # Save the df to excel
        df.to_excel(Flame.outpath)

        return df

    def view_pimage(self):
        """Method to view the Thresh Image before passing to edge dectector"""

        img = cv2.imread(Flame.imagepath)
        cv2.imshow('Processed Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return 0


if __name__ == '__main__':
    path = r'E:\Github\Flame-Speed-Tool\bin\test.avi'

    points = Crop(path=path, out=r'E:\Github\Flame-Speed-Tool').crop_video()
    # cls1 = Flame(path)
    # cls1.process(breaks=4, filter_size=[50, 50, 50, 50], thresh_val=[25, 50, 80, 75], crop_points=points,
    #              flow_right=True)
    #
    #
    # def show(val):
    #     plt.imshow(cv2.imread(val))
    #     plt.show()
