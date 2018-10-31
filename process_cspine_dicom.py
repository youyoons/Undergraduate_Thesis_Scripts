import os
import sys
import shutil
from shutil import copyfile
import settings
#import utils
import cv2
import SimpleITK
import numpy as np
import pydicom
import math
import xml.etree.ElementTree as ET
import pandas
import traceback

def save_log(path,entries,exclusions):

    df = pandas.DataFrame(entries, columns=["Study","Series","File"]+exclusions)
    df.to_csv(path, index=False)


def inspect_scan(scan_path):
    
    scan = pydicom.read_file(scan_path) 
    print(scan.dir())
    print(scan)
    
def process_study(scans,file_names,base_dir,patient,exclusions,log_path):

    all_series = {} #stores all the series (studies done on a single patient)
    
    #Study is the patient that is being studied
    study = scans[0].StudyInstanceUID    

    print("Processing study: {}".format(study))

    assert len(scans)==len(file_names), "scans list and file_names list have diff lengths"
 
    #identify all series (different patient studies) in the study (patient)
    for index,scan in enumerate(scans): #enumerate starts from 0
        series = scan.SeriesInstanceUID
        instance = int(scan.InstanceNumber) #needed as a unique identifier (multiple DICOM files have same series)
        series_desc = scan.SeriesDescription
        #print(series_desc)

        if series not in all_series:
            all_series[series] = [[instance,file_names[index],series_desc]]
        else:
            all_series[series].append([instance,file_names[index],series_desc])

    #create directories for different series
    study_path_raw = os.path.join("/home/youy/Documents/Spine/RawData_y",study)
 
    study_path_proc = os.path.join("/home/youy/Documents/Spine/ProcessedData_y",study)

    [os.mkdir(path) for path in [study_path_raw,study_path_proc] if not os.path.exists(path)]

    # Stores the series description that fits the CSpine scans we want
    valid_series_desc = ["C SPINE BONE SAG 1.2", "C-SPINE BONE SAG 1.2", "C-SPINE SAG", "C-SPINE SPINE BONE SAG",
                         "CSPINE SAG", "SAG C-SPINE BONE"]

    for series in all_series:
        if (len(all_series[series])) > 1 and any(all_series[series][0][2] in s for s in valid_series_desc):
            if not os.path.exists(os.path.join(study_path_raw,series)):
                os.mkdir(os.path.join(study_path_raw,series))

            if not os.path.exists(os.path.join(study_path_proc,series)):
                os.mkdir(os.path.join(study_path_proc,series))


    #move raw files into directories
    for series in all_series:
        instances = all_series[series]

        if (len(all_series[series])) > 1 and any(all_series[series][0][2] in s for s in valid_series_desc):
            for scan in all_series[series]:
                src = scan[1] #DICOM file location
                dst = os.path.join(study_path_raw,series,str(scan[0])+".dcm")

                if not os.path.exists(dst):
                    #shutil.move(src,dst)
                    copyfile(src,dst)
                else:
                    dst = os.path.join(study_path_raw,series,str(scan[0])+".dcm")

    #for linking log entries
    log_entries = []

    #dicom to dicom and save
    for series in all_series:
        if (len(all_series[series])) > 1 and any(all_series[series][0][2] in s for s in valid_series_desc):
            series_dir = os.path.join(study_path_raw,series)
            series_files = [os.path.join(series_dir,file_name) for file_name in os.listdir(series_dir)]

            if len(series_files)<2:
                print("Series {} has less than two files".format(series))
                continue

            series_scans = [pydicom.read_file(file_name) for file_name in series_files]
            series_scans.sort(key = lambda x: float(x.InstanceNumber))

            try:
                slice_thickness = np.abs(series_scans[0].ImagePositionPatient[2] - series_scans[1].ImagePositionPatient[2])
            except:
                try:
                    slice_thickness = np.abs(series_scans[0].SliceLocation - series_scans[1].SliceLocation)
                except:
                    slice_thickness = np.abs(0)

            for s in series_scans:
                s.SliceThickness = slice_thickness
 
            instance_nums = [scan.InstanceNumber for scan in series_scans]
        
            for index,scan in enumerate(series_scans):
                file_name = "scan_%s_%s" % (str(instance_nums[index]).rjust(4, '0'),".dcm")
                log_ids = [study,series,file_name]
            
                #saving path of anonymized one to proc directory
                save_path = os.path.join(study_path_proc,series,file_name)
            
                all_attributes = scan.dir()
                log_attr = []
            
                for attr in all_attributes:
                    if attr in exclusions:
                        tag = scan.data_element(attr).tag
                        value = scan[tag].value
                        log_attr.append(value)
                        del scan[tag]
                    
                print("Saving processed study to {}".format(save_path))
            
                #saving to processed location
                scan.save_as(save_path)
            
                #adding to log_entries, which will be used for the linking log
                log_entries.append(log_ids+log_attr)
             
            print("Series {} complete".format(series))

    
    return log_entries
 

def process_scans(data_dir="/media/ubuntu/cryptscratch/scratch/youy/Data/Spine/"):
    dicom_exclusions = ['AccessionNumber','AdditionalPatientHistory','NameOfPhysiciansReadingStudy','OtherPatientIDs','OtherPatientIDsSequence','PatientBirthDate','PatientID','PatientName']    
    
    raw_data_dir = "/home/youy/Documents/Spine/batch1/"
    log_path = "/home/youy/Documents/Spine/linking_log.csv"

    patients = os.listdir(raw_data_dir)
    print(patients)
    
    log_entries = []
    
    for patient in patients:
        try:
            patient_path = os.path.join(raw_data_dir, patient)
            print("Looking at dicoms in {}".format(patient_path))
            dicom_filenames = os.listdir(patient_path)
            dicom_files = [file_name for file_name in dicom_filenames if file_name[-4:]==".dcm"]
            #print(dicom_files)
            
            slices = [pydicom.read_file(os.path.join(patient_path,s)) for s in dicom_files] 
            file_names = [os.path.join(patient_path,s) for s in dicom_files]
            #print(slices)
            
            log_line = process_study(slices,file_names,data_dir,patient,dicom_exclusions,log_path)   
            
            log_entries.extend(log_line)    
            print('-------------------------------------------')
       
        except Exception as e:
            print("Error with {} : {}".format(patient,e))
            traceback.print_exc()
            
    save_log(log_path,log_entries,dicom_exclusions)

if __name__=="__main__":
    process_scans()
   
