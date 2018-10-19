import os
import numpy as np
import pydicom
import argparse
import pickle

#Assumes that we are one level above the Study directory level
def get_accession_number(path):
    accession_number = []

    #Assumes path gives file path to batch
    studies = os.listdir(path)
    #print(studies)

    print("Number of Studies: ", len(studies))

    for study in studies:
        #print("On Study: ", study)
        series_path = os.path.join(path,study)

        series = os.listdir(series_path) 
        #print(len(series))
        #print(series)

        #Set first file to get accession numbers
        if ".dcm" in series[0]:
            first_file = series[0]
        else:
            if len(series) < 2:
                first_file = None
            else:
                first_file = series[1]

        if first_file != None:
            #print(first_file)
         
            scan = pydicom.read_file(os.path.join(series_path,first_file))
          
            #Only adding unique accession number, series descriptions            
            if scan.AccessionNumber not in accession_number:
                accession_number.append(scan.AccessionNumber)
        
        '''
        for series_file in series:
            if ".dcm" in series_file:
                #print(first_file)

                scan = pydicom.read_file(os.path.join(series_path,series_file))
          
                #Only adding unique accession number, series descriptions            
                if scan.AccessionNumber not in accession_number:
                    accession_number.append(scan.AccessionNumber)
    
                if scan.SeriesDescription not in series_description:
                    series_description.append(scan.SeriesDescription)
        '''
    return accession_number

def get_series_description(path):
    series_description = []

    #Assumes path gives file path to batch
    studies = os.listdir(path)

    print("Number of Studies: ", len(studies))

    sampled_studies = studies[0:len(studies)//100]
    for study in sampled_studies:
        #print("On Study: ", study)
        series_path = os.path.join(path,study)

        series = os.listdir(series_path) 

        for series_file in series:
            if ".dcm" in series_file:
                #print(first_file)

                scan = pydicom.read_file(os.path.join(series_path,series_file))
          
                #Only adding unique series descriptions            
                if scan.SeriesDescription not in series_description:
                    series_description.append(scan.SeriesDescription)

    return series_description

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Getting batch info')
    parser.add_argument("crypt", help='which crypt location')
    parser.add_argument("batch", help='which batch number')
      
    args = parser.parse_args()
 
    crypt = args.crypt
    batch = args.batch       
     
    if crypt == "cryptscratch":
        path = "/media/ubuntu/" + crypt + "/cspine_data/" + batch + "/Final/"
    else:
        path = "/media/ubuntu/" + crypt + "/cspine_data2/" + batch + "/Final/"

    acNum = get_accession_number(path)    
    print("Accession Numbers: ", acNum)
    print("Number of AcNum: ", len(acNum))
    

    serDesc = get_series_description(path)
    print("Series Descriptions: ", serDesc)
    #print(len(serDesc))
    
