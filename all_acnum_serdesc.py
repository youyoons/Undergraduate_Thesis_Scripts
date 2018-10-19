import os
import sys
import shutil
import settings
import utils
import cv2
import SimpleITK
import numpy as np
import pydicom
import math
import pandas

def get_acNum_serDesc(path):
    #Assumes path gives file path to batch
    studies = os.listdir(path)
    
    accession_number = []
    series_description = []
    
    for study in studies:
        series_path = os.path.join(path,study)
        
        series = os.listdir(series_path)[0] #Get the first dicom file in a series
        
        scan = pydicom.read_file(series)
        
        accession_number.append(scan.AccessionNumber)
        series_description.append(scan.SeriesDescription)
    
    return accession_number
    

if __name__=="__main__":
    path = "/media/ubuntu/cryptscratch/cspine_data/batch1/Final/"
    
    acNum,serDesc = get_acNum_seriesDesc(path)
    
    print("Accession Numbers: ", acNum)
    print("Series Descriptions: ", serDesc)