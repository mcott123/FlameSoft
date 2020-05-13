import numpy as np
import cv2 as cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd


class data(object):
    """
    The data class is process the numpy data stored in the text file.
    
    1. self.label= 'left'
       Tracks the flame speed direction
       
       left = flame is moving left on the screen
       right = flame is moving right on the screen
            
    2. self.pixel_val = 'max'
       Tracks the location of the flame
       
       'max' = maximum value in the flame direction
       
       'avg' = average of the all the datapoints for the flame location
    """

    def __init__(self, label='left', pixel_val='max', fps='3000'):

        self.label = label

        self.frames = 0

        self.pixel_val = pixel_val

        self.pixel_x_num = 0

        self.pixel_y_num = 0

        self.length_x = 24

        self.length_y = 12

        self.fps = 30

        self.dataframe = pd.DataFrame()

        self.heightVariable = self.length_y / 2

        self.lengthVariable = self.length_x / 2

        self.lengthvar = 0

        self.heightvar = 0

    def load_data(self, path):
        """
        The path variable is the location of the text file conating the data in numpy matrix
        """
        self.array = np.load(path)

        # Number of frames are
        self.frames = self.array.shape[0]

        # Number of Pixel in X and Y are

        self.pixel_x_num = self.array.shape[2]

        self.pixel_y_num = self.array.shape[1]

        return self.array

    def calc(self):

        self.loc_x = []
        self.loc_y = []

        """ Implementing the data collection 
        1. input the pixel value
        """

        for frame in range(self.frames):
            """
            Calculating the flame speed from arguments
            max= maximum pixel distance
            """
            canny_edge = self.array[frame]

            flame_loc = np.where(canny_edge == 255)

            ####################################################################
            if self.label == 'right':
                ##############################################
                if self.pixel_val == 'max':

                    if len(flame_loc[0]) > 1 and len(flame_loc[1]) > 1:
                        flame_loc_x = max(flame_loc[1])

                        flame_loc_y = self.pixel_y_num - min(flame_loc[0])

                    else:
                        flame_loc_x = None

                        flame_loc_y = None

                    self.loc_x.append(flame_loc_x)
                    self.loc_y.append(flame_loc_y)

                    ###############################################
                elif self.pixel_val == 'avg':

                    if len(flame_loc[0]) > 1 and len(flame_loc[1]) > 1:
                        flame_loc_x = np.nanmean(flame_loc[1])

                        flame_loc_y = self.pixel_y_num - np.nanmean(flame_loc[0])

                    else:
                        flame_loc_x = None

                        flame_loc_y = None

                    self.loc_x.append(flame_loc_x)
                    self.loc_y.append(flame_loc_y)
                    ###############################################

                elif self.pixel_val == 'height':

                    ##varheight = pixel per feet accross height
                    pixeldensity_height = self.pixel_y_num / self.length_y
                    self.heightvar = int(pixeldensity_height * self.heightVariable)

                    ##varlength = pixel per feet accross length
                    pixeldensity_length = self.pixel_x_num / self.length_x
                    self.lengthvar = int(pixeldensity_length * self.lengthVariable)

                    flame_loc_x = np.average(np.where(canny_edge[self.heightvar] != 0))
                    flame_loc_y = self.pixel_y_num - np.average(np.where(np.transpose(canny_edge)[self.lengthvar] != 0))

                    self.loc_x.append(flame_loc_x)
                    self.loc_y.append(flame_loc_y)

            if self.label == 'left':
                ########################################################################
                if self.pixel_val == 'max':

                    if len(flame_loc[0]) > 1 and len(flame_loc[1]) > 1:
                        flame_loc_x = self.pixel_x_num - min(flame_loc[1])

                        flame_loc_y = self.pixel_y_num - min(flame_loc[0])

                    else:
                        flame_loc_x = None

                        flame_loc_y = None

                    self.loc_x.append(flame_loc_x)
                    self.loc_y.append(flame_loc_y)

                    ##########################################################################
                elif self.pixel_val == 'avg':

                    if len(flame_loc[0]) > 1 and len(flame_loc[1]) > 1:
                        flame_loc_x = np.nanmean(flame_loc[1])

                        flame_loc_y = self.pixel_y_num - np.nanmean(flame_loc[0])

                    else:
                        flame_loc_x = None

                        flame_loc_y = None

                    self.loc_x.append(flame_loc_x)
                    self.loc_y.append(flame_loc_y)


                elif self.pixel_val == 'height':

                    ##varheight = pixel per feet accross height
                    pixeldensity_height = self.pixel_y_num / self.length_y
                    self.heightvar = int(pixeldensity_height * self.heightVariable)

                    ##varlength = pixel per feet accross length
                    pixeldensity_length = self.pixel_x_num / self.length_x
                    self.lengthvar = int(pixeldensity_length * self.lengthVariable)

                    flame_loc_x = self.pixel_x_num - np.average(np.where(canny_edge[self.heightvar] != 0))
                    flame_loc_y = self.pixel_y_num - np.average(np.where(np.transpose(canny_edge)[self.lengthvar] != 0))

                    self.loc_x.append(flame_loc_x)
                    self.loc_y.append(flame_loc_y)

        return self.loc_x, self.loc_y

    def dataFrame(self):

        """Making a Dataframe
        """
        dataframe = pd.DataFrame(list(zip(self.loc_x, self.loc_y)), columns=['X pixel', 'Y pixel'])

        # Pixel length in x and y direction

        dx = self.length_x / self.pixel_x_num
        dy = self.length_y / self.pixel_y_num

        # Multiply the pixle number to get the length location of flame front

        dataframe['Y loc'] = dataframe['Y pixel'] * dy
        dataframe['X loc'] = dataframe['X pixel'] * dx
        dataframe['Time'] = dataframe.index / self.fps

        self.dataframe = dataframe

        return self.dataframe
