# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:34:10 2023

@author: 7009291
"""
# import packages
import os
import pandas as pd
import numpy as np
from classes.preprocessing import preprocessing
#from classes.video_processing import video_processing
#from classes.image_processing import image_processing
from classes.event_extraction import event_extraction
from classes.pupil_prediction import pupil_prediction
from classes.interactive_plot import interactive_plot
from scipy.interpolate import PchipInterpolator

import pickle
import cv2
import threading
from threading import *
from PIL import Image, ImageTk
import sys
import logging

import psutil
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import time
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button,TextBox,CheckButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
########################################################################################################################################################
#####################################################Information entered by the user####################################################################
########################################################################################################################################################
# Directories
initialDir = "D:\\Github\\Open-DPSM" # This should be the directory of the Open-DPSM

dataDir =  "D:\\Github\\Data\\Space Transit w- EEG\\Test 11-04-2024 - 2nd test with EEG\\result\\" # This should be the folder saving the eyetracking data and the video data 
## eyetracking data:
###- should have four columns in the order as: time stamps, gaze position x, gaze position y, pupil size
###- time stamps should be in seconds (not miliseconds). If not, please convert it to seconds
subjectFileName = "left_coordinates_train.csv" # name of the subject (data file should be contained by "dataDir") [Comment out this line if no eyetracking data]
subjectFileName_val = "left_coordinates_val.csv"
# subjectFileName_test = "left_coordinates_test2.csv"
# subjectFileName_test2 = "left_coordinates_test.csv"
## video data
### Format can be used: .mp4,.avi,.mkv,.mwv,.mov,.flv,.webm (other format can also be used as long as it can be read by cv2)
movieName =  "left_train.mp4" # name of the movie (data file should be contained by "dataDir")
movieName_val = "left_val.mp4"
# movieName_test = "left_test2.mp4"
# movieName_test2 = "left_test.mp4"

## If the movie and the eyetracking data are not the same, what to do?
stretchToMatch = True # True: stretch the eyelinkdata to match the movie data; False: cut whichever is longer
### maximum luminance of the screen (luminance when white is showed on screen)
maxlum = 115

## The following information is only relevant if eyetracking data is available
### What is the resolution of eyetracking data 
eyetracking_height = 886
eyetracking_width = 996
### What is the video (showed on the screen) resolution (respective to eyetracking resolution).
# *Note that it is not the resolution in the video file.* For example, if the resolution of the eye-tracking data is 1000x500 and the physical height and width of the video displayed is half of the physical height and width of the screen, then videoRealHeight & videoRealHeight should be 500 and 250
  
videoRealHeight = 886
videoRealWidth = 996

## if video resolution is not the same as eyetracking resolution, what color is the screen covered with? (enter rgb value, E.g. r,b,g = 0 means black)
screenBgColorR = 0
screenBgColorG = 0
screenBgColorB = 0
### Do you want to save:
# - model evaluation & paramters
saveParams = True
# - data used for modeling
saveData = True



###############

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:15:02 2023

@author: 7009291
"""

# Pre-determined parameters: Don't change unless absolutely sure
# boolean indicating whether or not to show frame-by-frame of each video with analysis results
showVideoFrames = True

# boolean indicating whether or not to skip feature extraction of already analyzed videos.
# If skipped, then video information is loaded (from pickle file with same name as video).
# If not skipped, then existing pickle files (one per video) are overwritten.
skipAlrAnFiles = True

# array with multiple levels [2, 4, 8], with each number indicating the number of vertical image parts (number of horizontal parts will be in ratio with vertical); the more levels, the longer the analysis
nVertMatPartsPerLevel = [3, 6]  # [4, 8, 16, 32]
# computer aspect ratio from the width and height
aspectRatio = 0.75 #videoRealWidth/videoRealHeight
imageSector = "6x8" # number of visual field regions (used for naming the new visual feature)
# integer indicating number of subsequent frames to calculate the change in features at an image part.
# it is recommended to set this number such that 100ms is between the compared frames.
# e.g. for a video with 24fps (50ms between subsequent frames), the variable should be set at 2.
# we are using 60fps video, so we calculate the frames needed to be skipped to have 100ms between frames
nFramesSeqImageDiff = int(100/(1000/60))

# string indicating color space in which a 3D vector is calculated between frames
# Options: 'RGB', 'LAB', 'HSV', 'HLS', 'LUV'
colorSpace = "LAB"

# list with strings indicating which features to analyze
# If colorSpace = 'LAB', options: ["Luminance","Red-Green","Blue-Yellow","Lum-RG","Lum-BY","Hue-Sat","LAB"]
# If colorSpace = 'HSV', options: ["Hue","Saturation","Luminance","Hue-Sat","Hue-Lum","Sat-Lum","HSV"]
featuresOfInterest = [
    "Luminance",
    # "Red-Green",
    # "Blue-Yellow",
    # "Lum-RG",
    # "Lum-BY",
    # "Hue-Sat",
    # "LAB",
]

# monitor gamma factor
# for conversion from pixel values to luminance values in cd/m2
# Set to zero for no conversion
scrGamFac = 2.2 #TODO, we should figure out the gamma factor of the screen
# What is the ratio between gaze-centered coordinate system and the screen
A = 2
# Number of movie frames skipped at the beginning of the movie
skipNFirstFrame = 0
# gaze-contingent
gazecentered = True
############pupil prediction parameters##############
# Response function type
RF = "HL"
# same regional weights for luminance and contrast
sameWeightFeature = False 
# Basinhopping or minimizing
useBH = False
# iteration number for basinhopping
niter = 5

# This is the indicator that the app is not used.
useApp = False
# chdir
os.chdir(initialDir)

class PupilModeling:
    def __init__(self):
        self.prepObj = None
        self.eeObj = None
        self.isTraining = True
        self.modelObj = None
        self.plotObj = None
        self.magnPerImPart = None
        self.magnPerIm = None

    def removeBlinks(self, blinkDetectionThreshold=[4, 2], dilutionSize=[4, 8], consecutiveRemovalPeriod=4, plotBlinkRemoveResult=False):
        """
        Process the DataFrame self.df_eyetracking to remove blinks.
        The DataFrame should have columns: timestamps, gazeX, gazeY, pupil diameters.
        """
        # Extract data from DataFrame
        timeStamps = self.df_eyetracking.iloc[:, 0].values
        pupildata = self.df_eyetracking.iloc[:, 3].values
        gazex = self.df_eyetracking.iloc[:, 1].values
        gazey = self.df_eyetracking.iloc[:, 2].values

        # Convert to float32
        timeStamps = np.array(timeStamps, dtype="float32")
        pupildata = np.array(pupildata, dtype="float32")
        gazex = np.array(gazex, dtype="float32")
        gazey = np.array(gazey, dtype="float32")

        # Filter out blink periods (blink periods contain value 0; set at 2 just in case)
        pdata = pupildata.copy()
        gazexdata = gazex.copy()
        gazeydata = gazey.copy()
        pdata[pdata < 2] = np.nan

        # Filter out blinks based on speed changes
        pdiff = np.diff(pdata)  # difference between consecutive pupil sizes

        # Create blinkspeed threshold 4SD below the mean
        blinkSpeedThreshold = np.nanmean(pdiff) - (blinkDetectionThreshold[0] * np.nanstd(pdiff))

        # Create blinkspeed threshold 2SD above the mean
        blinkSpeedThreshold2 = np.nanmean(pdiff) + (blinkDetectionThreshold[1] * np.nanstd(pdiff))

        # Blink window containing minimum and maximum value
        blinkWindow = [-dilutionSize[0], dilutionSize[1]]
        blinkWindow2 = [-dilutionSize[1], dilutionSize[0]]

        blinkIdx = np.where(pdiff < blinkSpeedThreshold)[0]  # find where the pdiff is smaller than the lower blinkspeed threshold
        blinkIdx = blinkIdx[np.where(np.diff(blinkIdx) > consecutiveRemovalPeriod)[0]]

        blinkIdx2 = np.where(pdiff > blinkSpeedThreshold2)[0]  # find where the pdiff is larger than the upper blinkspeed threshold
        blinkIdx2 = blinkIdx2[np.where(np.diff(blinkIdx2) > consecutiveRemovalPeriod)[0]]

        # Remove blink segments
        for bl in blinkIdx:
            pdata[np.arange(bl + blinkWindow[0], bl + blinkWindow[1])] = np.nan
            pdiff[np.arange(bl + blinkWindow[0], bl + blinkWindow[1])] = np.nan

        for bl in blinkIdx2:
            pdata[np.arange(bl + blinkWindow2[0], bl + blinkWindow2[1])] = np.nan
            pdiff[np.arange(bl + blinkWindow2[0], bl + blinkWindow2[1])] = np.nan

        # Interpolate blink periods
        missDataIdx = np.where(~np.isfinite(pdata))[0]
        corrDataIdx = np.where(np.isfinite(pdata))[0]
        ## PCHIP interpolation
        pdata_beforeInterpo = pdata.copy()
        # Remove gazex and gazey data if pdata were identified as NaN
        gazexdata[~np.isfinite(pdata)] = np.nan
        gazeydata[~np.isfinite(pdata)] = np.nan
        # Interpolate Pupil and Gaze data
        pdata[missDataIdx] = PchipInterpolator(timeStamps[corrDataIdx], pdata[corrDataIdx])(timeStamps[missDataIdx], extrapolate=False)
        gazexdata[missDataIdx] = PchipInterpolator(timeStamps[corrDataIdx], gazexdata[corrDataIdx])(timeStamps[missDataIdx], extrapolate=False)
        gazeydata[missDataIdx] = PchipInterpolator(timeStamps[corrDataIdx], gazeydata[corrDataIdx])(timeStamps[missDataIdx], extrapolate=False)

        # # if there are nan at the beginning, fill them with the first non-nan value
        # firstNonNan = np.where(np.isfinite(pdata))[0][0]
        # pdata[:firstNonNan] = pdata[firstNonNan]
        # gazexdata[:firstNonNan] = gazexdata[firstNonNan]
        # gazeydata[:firstNonNan] = gazeydata[firstNonNan]

        self.df_eyetracking.iloc[:, 3] = pdata
        self.df_eyetracking.iloc[:, 1] = gazexdata
        self.df_eyetracking.iloc[:, 2] = gazeydata

    def preprocessing(self, movieName, subjectFileName):
        os.chdir(initialDir)
        self.movieName = movieName
        self.subjectFileName = subjectFileName
        # read video data and check information
        self.prepObj = preprocessing()
        self.filename_movie = dataDir +"\\" + movieName
        self.prepObj.videoFileName = self.filename_movie
        self.prepObj.preprocessingVideo()
        self.video_nFrame = self.prepObj.vidInfo['frameN_end']
        self.video_height = self.prepObj.vidInfo['height']
        self.video_width = self.prepObj.vidInfo['width']
        self.video_ratio = self.video_height / self.video_width 
        self.video_duration = self.prepObj.vidInfo['duration_end']
        self.video_fps = self.prepObj.vidInfo['fps_end']
        print(f"Video number of frame: {self.video_nFrame}")
        print(f"Video height x width: {self.video_height}x{self.video_width}; aspect ratio (width:height): {1/self.video_ratio}")
        print(f"Video duration: {self.video_duration}")
        print(f"Video frame rate: {self.video_fps}")
        self.movieName = movieName.split(".")[0]

        filename_csv = dataDir + "\\" + subjectFileName
        # read eyetracking data and check information
        self.df_eyetracking = pd.read_csv(filename_csv, index_col=0, header = 0)
        self.removeBlinks()
        # change the beginning as 0s
        self.df_eyetracking.iloc[:,0] = self.df_eyetracking.iloc[:,0]-self.df_eyetracking.iloc[0,0]
        self.eyetracking_duration = self.df_eyetracking.iloc[-1,0]
        self.eyetracking_nSample = self.df_eyetracking.shape[0]
        self.eyetracking_samplingrate = int(1/(self.eyetracking_duration/self.eyetracking_nSample))
        self.subjectName = subjectFileName.split(".")[0]

        print(f"Eyetracking data duration: {self.eyetracking_duration} seconds")
        print(f"Eyetracking data sampling rate: {self.eyetracking_samplingrate} Hz")

        # check if video and the eyetracking data have the same ratio
        if videoRealHeight == eyetracking_height and videoRealWidth ==eyetracking_width:
            self.videoScreenSameRatio = True 
            self.videoStretched = True
        elif videoRealHeight == eyetracking_height or videoRealWidth ==eyetracking_width:
            self.videoScreenSameRatio = False
            self.videoStretched = True
        else:
            self.videoScreenSameRatio = False
            self.videoStretched = False

    def visual_event_extraction(self, visual_event_file = ''):
        os.chdir(dataDir)
        foldername = "Visual events"
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        os.chdir(foldername)

        if os.path.exists(visual_event_file):
            with open(visual_event_file, "rb") as handle:
                self.vidInfo, self.timeStamps, self.magnPerImPart, self.magnPerIm = pickle.load(handle)
                handle.close()
            return
        
        
        # name the feature extracted pickle:
        picklename = self.movieName + "_"+ self.subjectName + "_VF_" + colorSpace + "_" + str(maxlum) + "_" + imageSector + ".pickle"

        if os.path.exists(picklename):
            with open(picklename, "rb") as handle:
                self.vidInfo, self.timeStamps, self.magnPerImPart, self.magnPerIm = pickle.load(handle)
                handle.close()
            return

        # feature extraction class
        self.eeObj = event_extraction()

        # load some data and parameters
        self.eeObj.video_duration = self.video_duration
        self.eeObj.video_fps = self.video_fps
        self.eeObj.stretchToMatch = stretchToMatch
        self.eeObj.subject = self.subjectName
        self.eeObj.movieNum = movieName
        self.eeObj.picklename = picklename
        self.eeObj.filename_movie = self.filename_movie
        self.eeObj.setNBufFrames(nFramesSeqImageDiff + 1)
        self.eeObj.imCompFeatures = True  # creates: imageObj.vectorMagnFrame
        self.eeObj.showVideoFrames = showVideoFrames
        self.eeObj.imColSpaceConv = colorSpace
        self.eeObj.gazecentered = gazecentered
        self.eeObj.nVertMatPartsPerLevel = nVertMatPartsPerLevel  # [4, 8, 16, 32]
        self.eeObj.aspectRatio = aspectRatio 
        self.eeObj.imageSector = imageSector
        self.eeObj.nFramesSeqImageDiff = nFramesSeqImageDiff
        self.eeObj.selectFeatures = featuresOfInterest
        self.eeObj.scrGamFac = scrGamFac
        self.eeObj.A = A
        self.eeObj.maxlum = maxlum
        self.eeObj.useApp = useApp
        self.eeObj.videoScreenSameRatio = self.videoScreenSameRatio 
        self.eeObj.videoStretched = self.videoStretched    
        self.eeObj.vidInfo = self.prepObj.vidInfo # extract vidInfo from preprocessing object

        # process eyetracking data
        if gazecentered: # if there is eyetracking data, do gaze-contingent visual events extraction
            self.eeObj.eyetracking_duration = self.eyetracking_duration
            self.eeObj.eyetracking_height = eyetracking_height
            self.eeObj.eyetracking_width = eyetracking_width
            self.eeObj.eyetracking_samplingrate = self.eyetracking_samplingrate
            self.eeObj.videoRealHeight = videoRealHeight
            self.eeObj.videoRealWidth = videoRealWidth
            self.eeObj.screenBgColorR = screenBgColorR
            self.eeObj.screenBgColorG = screenBgColorG
            self.eeObj.screenBgColorB = screenBgColorB
            timeStampsSec = np.array(self.df_eyetracking.iloc[:,0])
            gazexdata = np.array(self.df_eyetracking.iloc[:,1])
            gazeydata = np.array(self.df_eyetracking.iloc[:,2])
            # resample the eytracking data to match the video sampling rate
            self.eeObj.sampledTimeStamps_featureExtraction = self.eeObj.prepare_sampleData(timeStampsSec, self.video_nFrame)
            self.eeObj.sampledgazexData_featureExtraction = self.eeObj.prepare_sampleData(gazexdata, self.video_nFrame)
            self.eeObj.sampledgazeyData_featureExtraction = self.eeObj.prepare_sampleData(gazeydata, self.video_nFrame)
        # start feature extraction: this can take a while. The extracted features will be saved in folder "Visual events"
        self.eeObj.event_extraction()

        self.vidInfo = self.eeObj.vidInfo
        self.timeStamps = self.eeObj.sampledTimeStamps_featureExtraction
        self.magnPerImPart = self.eeObj.magnPerImPart
        self.magnPerIm = self.eeObj.magnPerIm

    def pupil_modeling(self, subjectName = ''):
        os.chdir(dataDir)
        os.chdir("Visual events")

        if not self.magnPerImPart or not self.magnPerIm:
            print("No visual event data found. Run visual event extraction first.")
            self.visual_event_extraction()
        
        foldername = "Modeling result"
        os.chdir(dataDir) 
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        os.chdir(foldername)

        #Create dictionaries to save results
        if os.path.exists("modelDataDict.pickle"):
            with open("modelDataDict.pickle", "rb") as handle:
                self.modelDataDict = pickle.load(handle)
                handle.close() 
        else:
            self.modelDataDict = {}
                
        if os.path.exists("modelResultDict.pickle"):
            with open("modelResultDict.pickle", "rb") as handle:
                self.modelResultDict = pickle.load(handle)
                handle.close() 
        else:
            self.modelResultDict = {}

        self.modelObj = pupil_prediction()

        self.modelObj.subject = self.subjectName
        self.modelObj.movie = self.movieName
        self.modelObj.sameWeightFeature = sameWeightFeature
        self.modelObj.RF = RF 
        self.modelObj.skipNFirstFrame = skipNFirstFrame 
        self.modelObj.useBH = useBH
        self.modelObj.niter = niter
        self.modelObj.magnPerImPart= self.magnPerImPart
        self.modelObj.useApp = useApp
        self.modelObj.stretchToMatch = stretchToMatch
        self.modelObj.video_duration = self.video_duration
        self.modelObj.eyetracking_duration = self.eyetracking_duration
        self.modelObj.video_fps = self.video_fps
        self.modelObj.nFramesSeqImageDiff = nFramesSeqImageDiff

        # load eyetracking data
        self.modelObj.useEtData = True
        timeStampsSec = np.array(self.df_eyetracking.iloc[:,0])
        gazexdata = np.array(self.df_eyetracking.iloc[:,1])
        gazeydata = np.array(self.df_eyetracking.iloc[:,2])
        pupildata = np.array(self.df_eyetracking.iloc[:,3])

        # downsampling the eyetracking data
        self.modelObj.eyetracking_samplingrate = self.eyetracking_samplingrate
        self.modelObj.sampledTimeStamps  =self.modelObj.prepare_sampleData(timeStampsSec,self.video_nFrame)
        self.modelObj.sampledgazexData =self.modelObj.prepare_sampleData(gazexdata,self.video_nFrame)
        self.modelObj.sampledgazeyData=self.modelObj.prepare_sampleData(gazeydata,self.video_nFrame)
        self.modelObj.sampledpupilData=self.modelObj.prepare_sampleData(pupildata,self.video_nFrame)
        self.modelObj.sampledFps = 1/(self.modelObj.sampledTimeStamps[-1]/(len(self.modelObj.sampledTimeStamps)))

        self.modelObj.sampledTimeStamps = self.modelObj.synchronize(self.modelObj.sampledTimeStamps)
        self.modelObj.sampledgazexData = self.modelObj.synchronize(self.modelObj.sampledgazexData)
        self.modelObj.sampledgazeyData = self.modelObj.synchronize(self.modelObj.sampledgazeyData)
        self.modelObj.sampledpupilData = self.modelObj.synchronize(self.modelObj.sampledpupilData)
        # save the data for undo zscore
        # self.pupil_mean = np.nanmean(self.modelObj.sampledpupilData)
        # self.pupil_std = np.nanstd(self.modelObj.sampledpupilData)
        self.modelObj.sampledpupilData= self.modelObj.zscore(self.modelObj.sampledpupilData)

        self.modelObj.modelDataDict = self.modelDataDict
        self.modelObj.modelResultDict = self.modelResultDict
        # 
        if self.isTraining:
            self.modelObj.pupil_prediction()
        else:
            params = self.modelResultDict[self.subjectName if subjectName == '' else subjectName]["modelContrast"]['parameters']
            self.modelObj.modelContrast(params)
            r,p = self.modelObj.correlation(self.modelObj.y_pred,  self.modelObj.sampledpupilData)
            rmse = self.modelObj.root_mean_of_squares_modelContrast(params)
            self.modelObj.r = r
            self.modelObj.rmse = rmse

        sampledTimeStamps = self.modelObj.sampledTimeStamps
        sampledpupilData = self.modelObj.sampledpupilData
        sampledFps = self.modelObj.sampledFps
        ##################################
        # save model results
        if saveParams and self.isTraining:
            foldername = "csv results"
            os.chdir(dataDir) 
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            os.chdir(foldername)
            params = self.modelResultDict[self.subjectName]["modelContrast"]["parameters"]
            paramNames = self.modelResultDict[self.subjectName]['modelContrast']['parametersNames']
            if RF == "HL":
                # paramNames = ["r",'rmse',"n_luminance", "tmax_luminance", "n_contrast", "tmax_contrast", "weight_contrast", "regional_weight1","regional_weight2","regional_weight3","regional_weight4","regional_weight5","regional_weight6"]
                # params = np.insert(params,5,1)
                # prepend the r and rmse at the front
                paramNames = ['r', 'rmse'] + paramNames
                params = np.insert(params,0,self.modelObj.r)
                params = np.insert(params,1,self.modelObj.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.subjectName}_parameters.csv")    
            elif RF == "KB":
                # paramNames = ["r",'rmse',"theta_luminance", "k_luminance", "theta_contrast", "k_contrast", "weight_contrast", "regional_weight1","regional_weight2","regional_weight3","regional_weight4","regional_weight5","regional_weight6"]
                # params = np.insert(params,5,1)
                paramNames = ['r', 'rmse'] + paramNames
                params = np.insert(params,0,self.modelObj.r)
                params = np.insert(params,1,self.modelObj.rmse)
                df = pd.DataFrame(np.vstack([paramNames,params]).T)
                df.columns = ["parameterName", "value"]
                df.to_csv(f"{self.subjectName}_parameters.csv")
        # save modeling data
        if saveData:
            foldername = "csv results"
            os.chdir(dataDir) 
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            os.chdir(foldername)
            y_pred = self.modelObj.y_pred
            lumConv = self.modelObj.lumConv
            contrastConv = self.modelObj.contrastConv
            sampledpupilData_z = (sampledpupilData -np.nanmean(sampledpupilData)) /np.nanstd(sampledpupilData)
            df = pd.DataFrame(np.vstack([sampledTimeStamps,sampledpupilData, y_pred,lumConv,contrastConv]).T)
            df.columns = ["timeStamps", "Actual pupil (z)", "Predicted pupil (z)", "Predicted pupil - luminance (z)", "Predicted pupil - contrast (z)"]
            
            df.to_csv(f"{self.subjectName}_modelPrediction.csv")

        return self.modelObj.r, self.modelObj.rmse, sampledTimeStamps, sampledpupilData, self.modelObj.y_pred, self.modelObj.lumConv, self.modelObj.contrastConv

    def plot_result(self, y_pred, lumConv, contrastConv):
        # making plot
        # This step have to be done after pupil prediction
        self.plotObj = interactive_plot()
        # subject and movie to plot
        self.plotObj.subjectName = self.subjectName
        self.plotObj.movieName = movieName
        # other parameters
        self.plotObj.useApp = useApp
        self.plotObj.dataDir = dataDir
        self.plotObj.filename_movie = self.prepObj.videoFileName
        self.plotObj.A = A
        self.plotObj.skipNFirstFrame = skipNFirstFrame
        self.plotObj.sampledFps = self.modelObj.sampledFps
        self.plotObj.eyetracking_height = eyetracking_height
        self.plotObj.eyetracking_width = eyetracking_width
        self.plotObj.videoRealHeight = videoRealHeight
        self.plotObj.videoRealWidth = videoRealWidth
        self.plotObj.screenBgColorR = screenBgColorR
        self.plotObj.screenBgColorG = screenBgColorG
        self.plotObj.screenBgColorB = screenBgColorB
        self.plotObj.videoScreenSameRatio = self.videoScreenSameRatio 
        self.plotObj.videoStretched = self.videoStretched
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        self.plotObj.plot_with_data(self.modelObj.sampledTimeStamps, self.modelObj.sampledpupilData, self.modelObj.sampledgazexData, self.modelObj.sampledgazeyData, self.modelObj.lumData, self.modelObj.contrastData, y_pred, lumConv, contrastConv)


##########################################Preprocessing: check the gaze data and the movie data######################

pupil_modeling = PupilModeling()
pupil_modeling.preprocessing(movieName=movieName, subjectFileName=subjectFileName)

# #%%############################################Visual events extraction##############################################
pupil_modeling.visual_event_extraction() # 'left_train_left_coordinates_train_VF_LAB_6x8.pickle'

# #%%############################################Pupil modeling##############################################
r, rmse, sampledTimeStamps, sampledpupilData, y_pred, lumConv, contrastConv = pupil_modeling.pupil_modeling()
print("training data length is:" + str(sampledTimeStamps[-1]))
print("r on training data is:" + str(r))
print("RMSE on training data is: " + str(rmse))

# #%%##################################### interactive plot##############################################
pupil_modeling.plot_result(y_pred=y_pred, lumConv=lumConv, contrastConv=contrastConv)


# ############################# Validation data #######################
pupil_modeling_val = PupilModeling()
pupil_modeling_val.preprocessing(movieName=movieName_val, subjectFileName=subjectFileName_val)
pupil_modeling_val.visual_event_extraction()
pupil_modeling_val.isTraining = False
r_val, rmse_val, sampledTimeStamps_val, sampledpupilData_val, y_pred_val, lumConv_val, contrastConv_val = pupil_modeling_val.pupil_modeling(subjectName=pupil_modeling.subjectName)
print("validation data length is:" + str(sampledTimeStamps_val[-1]))
print("r on validation data is:" + str(r_val))
print("RMSE on validation data is: " + str(rmse_val))
pupil_modeling_val.plot_result(y_pred=y_pred_val, lumConv=lumConv_val, contrastConv=contrastConv_val)

# pupil_modeling_test = PupilModeling()
# pupil_modeling_test.preprocessing(movieName=movieName_test, subjectFileName=subjectFileName_test)
# pupil_modeling_test.visual_event_extraction()
# pupil_modeling_test.isTraining = False
# r_test, rmse_test, sampledTimeStamps_test, sampledpupilData_test, y_pred_test, lumConv_test, contrastConv_test = pupil_modeling_test.pupil_modeling(subjectName=pupil_modeling.subjectName)
# print("test data length is:" + str(sampledTimeStamps_test[-1]))
# print("r on test data is:" + str(r_test))
# print("RMSE on test data is: " + str(rmse_test))
# pupil_modeling_test.plot_result(y_pred=y_pred_test, lumConv=lumConv_test, contrastConv=contrastConv_test)

# pupil_modeling_test2 = PupilModeling()
# pupil_modeling_test2.preprocessing(movieName=movieName_test2, subjectFileName=subjectFileName_test2)
# pupil_modeling_test2.visual_event_extraction()
# pupil_modeling_test2.isTraining = False
# r_test2, rmse_test2, sampledTimeStamps_test2, sampledpupilData_test2, y_pred_test2, lumConv_test2, contrastConv_test2 = pupil_modeling_test2.pupil_modeling(subjectName=pupil_modeling.subjectName)
# print("test data length is:" + str(sampledTimeStamps_test2[-1]))
# print("r on test data is:" + str(r_test2))
# print("RMSE on test data is: " + str(rmse_test2))
# pupil_modeling_test2.plot_result(y_pred=y_pred_test2, lumConv=lumConv_test2, contrastConv=contrastConv_test2)

# # test direct plot
# pupil_modeling = PupilModeling()
# pupil_modeling.preprocessing(movieName=movieName, subjectFileName=subjectFileName)
# pupil_modeling.isTraining = False
# r, rmse, sampledTimeStamps, sampledpupilData, y_pred, lumConv, contrastConv = pupil_modeling.pupil_modeling(subjectName=pupil_modeling.subjectName)
# print("training data length is:" + str(sampledTimeStamps[-1]))
# print("r on training data is:" + str(r))
# print("RMSE on training data is: " + str(rmse))
# pupil_modeling.plot_result(y_pred=y_pred, lumConv=lumConv, contrastConv=contrastConv)

# pupil_modeling_val = PupilModeling()
# pupil_modeling_val.preprocessing(movieName=movieName_val, subjectFileName=subjectFileName_val)
# pupil_modeling_val.isTraining = False
# r_val, rmse_val, sampledTimeStamps_val, sampledpupilData_val, y_pred_val, lumConv_val, contrastConv_val = pupil_modeling_val.pupil_modeling(subjectName=pupil_modeling.subjectName)
# print("validation data length is:" + str(sampledTimeStamps_val[-1]))
# print("r on validation data is:" + str(r_val))
# print("RMSE on validation data is: " + str(rmse_val))
# pupil_modeling_val.plot_result(y_pred=y_pred_val, lumConv=lumConv_val, contrastConv=contrastConv_val)

# pupil_modeling_test = PupilModeling()
# pupil_modeling_test.preprocessing(movieName=movieName_test, subjectFileName=subjectFileName_test)
# pupil_modeling_test.isTraining = False
# r_test, rmse_test, sampledTimeStamps_test, sampledpupilData_test, y_pred_test, lumConv_test, contrastConv_test = pupil_modeling_test.pupil_modeling(subjectName=pupil_modeling.subjectName)
# print("test data length is:" + str(sampledTimeStamps_test[-1]))
# print("r on test data is:" + str(r_test))
# print("RMSE on test data is: " + str(rmse_test))
# pupil_modeling_test.plot_result(y_pred=y_pred_test, lumConv=lumConv_test, contrastConv=contrastConv_test)


