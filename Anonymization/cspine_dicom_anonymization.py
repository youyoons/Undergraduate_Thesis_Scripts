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
    print(exclusions)
    df = pandas.DataFrame(entries, columns=["Study","Series","File"]+exclusions)
    df.to_csv(path, index=False)


def inspect_scan(scan_path):

    scan = pydicom.read_file(scan_path)
    print(scan.dir())
    print(scan)

def process_study(scans,file_names,patient,exclusions,log_path,path_raw,path_proc):

    all_series = {} #stores all the series (studies done on a single patient)

    #Study is the patient that is being studied
    study = scans[0].StudyInstanceUID
    accessnum = scans[0].AccessionNumber

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

    #Remove All Files/Directories Existing within RawData and ProcessedData
    #os.rmdir("/home/youy/Documents/Spine/RawData_y")
    #os.rmdir("/home/youy/Documents/Spine/ProcessedData_y")

    #create raw and processed directory if they don't exist
    [os.mkdir(path) for path in [path_raw,path_proc] if not os.path.exists(path)]

    #create directories for different series
    study_path_raw = os.path.join(path_raw,accessnum)
    study_path_proc = os.path.join(path_proc,accessnum)

    [os.mkdir(path) for path in [study_path_raw,study_path_proc] if not os.path.exists(path)]



    # Stores the series description that fits the CSpine scans we want
    valid_series_desc = ["C SPINE BONE SAG 1.2", "C-SPINE BONE SAG 1.2", "C-SPINE SAG", "C-SPINE SPINE BONE SAG","CSPINE SAG", "SAG C-SPINE BONE"]

    #for linking log entries
    log_entries = []

    for series in all_series:
        instances = all_series[series]

        #Create raw and processed series directories if they do not exist
        if (len(all_series[series])) > 1 and any(all_series[series][0][2] in s for s in valid_series_desc):
            #In terms of processed as we remove raw studies right after it is done
            print(os.path.join(study_path_proc,series))
            
            if not os.path.exists(os.path.join(study_path_proc,series)):
                os.mkdir(os.path.join(study_path_raw,series))
                os.mkdir(os.path.join(study_path_proc,series))

                #Copy over files into raw path
                for scan in all_series[series]:
                    src = scan[1] #DICOM file location
                    dst = os.path.join(study_path_raw,series,str(scan[0])+".dcm")

                    if not os.path.exists(dst):
                        #shutil.move(src,dst)
                        copyfile(src,dst)
                    else:
                        dst = os.path.join(study_path_raw,series,str(scan[0])+".dcm")

                #dicom to dicom and save

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
                    #print(all_attributes)
                
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



    directories = os.listdir("/home/youy/Documents/Spine/ProcessedData_y/")
    for i in directories:
        if len(os.listdir(os.path.join("/home/youy/Documents/Spine/ProcessedData_y/",i))) == 0:
            os.rmdir(os.path.join("/home/youy/Documents/Spine/ProcessedData_y/", i))

    #Remove raw directory path after each study
    shutil.rmtree(study_path_raw)    

    return log_entries


def process_scans(raw_data_dir,path_raw,path_proc,log_path):
    #Must be in alphabetical order
    dicom_exclusions = ['AccessionNumber','AdditionalPatientHistory','InstitutionName','NameOfPhysiciansReadingStudy','OtherPatientIDs','OtherPatientIDsSequence','PatientBirthDate','PatientID','PatientName','ReferringPhysicianName']

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

            log_line = process_study(slices,file_names,patient,dicom_exclusions,log_path,path_raw,path_proc)
            #print(log_line)
            log_entries.extend(log_line)
            print('-------------------------------------------')

        except Exception as e:
            print("Error with {} : {}".format(patient,e))
            traceback.print_exc()

    #print(dicom_exclusions)
    #print(log_entries[0:2])
    save_log(log_path,log_entries,dicom_exclusions)

if __name__=="__main__":

    raw_data_dir = "/home/youy/Documents/Spine/batch_test2/"   
    path_raw = "/home/youy/Documents/Spine/RawData_y"
    path_proc = "/home/youy/Documents/Spine/ProcessedData_y"
    log_path = "/home/youy/Documents/Spine/linking_log.csv"

    process_scans(raw_data_dir,path_raw,path_proc,log_path)

