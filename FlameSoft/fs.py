import cv2 as cv2
import numpy as np
import pandas as pd


class Crop(object):

    def __init__(self, path: str):
        """The crop class to get the pixel values for the cropped image
        path: string array to the path
        """
        self.path = path
        self.points = [(), ()]
        self.image = None

    def mouse_crop(self, event, x, y, flags, param):
        """Method to get the x and y locations on the image"""

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[0] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.points[1] = (x, y)

            # Draw the rectangle around the slected path
            cv2.rectangle(self.image, self.points[0], self.points[1], (255, 0, 0), 10)
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


class Flame(object):

    def __init__(self, path: str):
        self.path = path

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
        np.save(r'E:\Github\Flame-Speed-Tool\bin\test', ans)
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


class Data(object):

    def __init__(self, path: str):
        self.array = np.load(path)
        self.df = pd.DataFrame(self.array)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = r'E:\Github\Flame-Speed-Tool\bin\test.avi'

    cls = Crop(path).crop_video
    points = cls.points
    # points = [(930, 97), (1848, 666)]
    cls1 = Flame(path)
    cls1.process(breaks=4, filter_size=[50, 50, 50, 50], thresh_val=[25, 50, 80, 75],
                 crop_points=points, flow_right=True)


    def show(val):
        plt.imshow(val)
        plt.show()


    cls2 = Data(r'E:\Github\Flame-Speed-Tool\bin\test.npy')
    img = cls2.array
    show(img)
