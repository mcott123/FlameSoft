
# coding: utf-8

# In[1]:
from sys import version

print('Python Version=', version.split(' ')[0])

from os import environ

print('Environment =', environ['CONDA_DEFAULT_ENV'])

import sys
import pandas as pd
import PyQt5 as qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QFileDialog,QMessageBox
from PyQt5.QtCore import QCoreApplication
from gui import *
import fscode as fsc
import data_process as dapr
import numpy as np
import pyqtgraph as pg



class MyForm(qt.QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.browsePicture.clicked.connect(self.browsepic)
        self.ui.browseVideo.clicked.connect(self.browsevid)
        self.ui.processVideo.clicked.connect(self.process)
        self.ui.saveButton.clicked.connect(self.saveNum)
        self.ui.browseData.clicked.connect(self.browsedata)
        
        ##Data process Buttons linked
        self.ui.processButton.clicked.connect(self.dataProcess)
        self.ui.saveexcelButton.clicked.connect(self.saveDf)
        self.ui.plotButton.clicked.connect(self.plotFig)
        
        
        ##Define System variables for instances
        self.Points=[]
        self.vidFPS=0
        
        self.show()
        
    def browsepic(self):
        """Defines the image to be croped"""
        self.ui.picturePath.setText(str(QFileDialog.getOpenFileName()[0]))
    
    def browsevid(self):
        """Defines the input file for the video"""
        self.ui.videoPath.setText(str(QFileDialog.getOpenFileName()[0]))
    
    def browsedata(self):
        """Browse the data input file that is 3d numpy array"""
        self.ui.dataInput.setText(str(QFileDialog.getOpenFileName()[0]))
        
    def show_message(self,title,message):
        """Warning dialog box"""
        QMessageBox.warning(self,title,message)
    
    def process(self):
        
        """ Function to create an object with given parameters and process video. Returns the 3D numpy array """
        if self.ui.videoPath.text()=="" or self.ui.videoPath.text().endswith(('.avi','.wmv'))!=True :
            
            QMessageBox.warning(self,'Error in File Type','Please check video input to be .avi or .wmv extension',QMessageBox.Ok)
        
        elif self.ui.picturePath.text()==""and self.ui.previousBox.isChecked()==False:
            QMessageBox.warning(self,'Error in Input Image','Please input an image to crop video',QMessageBox.Ok)
        
        else:
            try:
                
                """Checking if the previous image box is checked"""
                if self.ui.previousBox.isChecked():
                    if self.Points==[]:
                        QMessageBox.warning(self,'Error Previous Image','Please Uncheck the use previous image and choose image path to crop',QMessageBox.Ok)
                    else:
                        img=self.Points
                    
                else:
                ##Get the croped image
                    img= fsc.fs(self.ui.picturePath.text()).area()

                    self.Points=img
                
                ##Video Process
                vidins=fsc.vid(self.ui.videoPath.text(),img,int(self.ui.filtersizeInput.text()),int(self.ui.pixelfilterInput.text()))
                
                vid=vidins.edge()

                self.matrix=vid
                self.vidFPS=round(vidins.insFPS)
                
            except ValueError:
                QMessageBox.warning(self,'Error in Process','Please check all the inputs',QMessageBox.Ok)

        
        
    def saveNum(self):
        """ Save the video processed array as numpy array with .npy extension ( 3D array )"""
        
        filename=str(QFileDialog.getSaveFileName()[0])
        np.save(filename,self.matrix)
        self.ui.dataInput.setText(filename+'.npy')
        
    
    def dataProcess(self):
        """Defines the funtion to process the 3D numpy array"""
        try:
            ## Arguments for process
            direc=self.ui.flameDirection.currentText().lower()
            handle=self.ui.dataHandle.currentText().lower()
            dataIn=self.ui.dataInput.text()
            
            ## Changing the attributes to user defined
            ins= dapr.data(pixel_val=handle,label=direc)
            ins.length_x=float(self.ui.lengthInput.text())
            ins.length_y=float(self.ui.heightInput.text())
            ins.fps=float(self.ui.fpsInput.text())
            ins.heightVariable=float(self.ui.flameheightInput.text())
            ins.lengthVariable=float(self.ui.flamelengthInput.text())
            
                
            ##Initiating OOP
            load=ins.load_data(dataIn)
            cal=ins.calc()
            dataframe=ins.dataFrame()
            self.datafr=dataframe
        
        except ValueError:
            QMessageBox.warning(self,'Error in Process','Please check all the inputs',QMessageBox.Ok)
        
        
    def saveDf(self):
        """Save the dataframe from dataProcess"""
        
        filename=str(QFileDialog.getSaveFileName()[0])
        writer=pd.ExcelWriter(filename+'.xlsx',engine='openpyxl') 
        self.datafr.to_excel(writer)
        writer.save()
    
    def plotFig(self):
        
        ##Always link the widget to self (no mistake)
        
        ##Create a general plot window
        self.window=pg.GraphicsWindow("Data plots for the Processed Data")
        
        ##Define plot windows and data to be added
        
        ##1
        plot1=self.window.addPlot(title=' Distance vs Time for X Direction')
        plot1.plot(self.datafr['Time'],self.datafr['X loc'])
        plot1.setLabel('left','Distance',units='feet')
        plot1.setLabel('bottom','Time',units='sec')
        
        ##2
        plot2=self.window.addPlot(title=' Distance vs Time for Y Direction')
        plot2.plot(self.datafr['Time'],self.datafr['Y loc'])
        plot2.setLabel('left','Distance',units='feet')
        plot2.setLabel('bottom','Time',units='sec')
        
    
    def check(self):
        pass


# In[2]:


if __name__=="__main__":
    app = qt.QtCore.QCoreApplication.instance()
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    if app is None:
        app = qt.QtWidgets.QApplication(sys.argv)
    ap=QtWidgets.QApplication([])    
    w = MyForm()
    w.show()
    ap.exec()

