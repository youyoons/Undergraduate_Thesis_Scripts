import os
import numpy as np
import pydicom

#Assumes that path is one level above the Study directory level
def get_acNum_serDesc(path):
    #Assumes path gives file path to batch
    studies = os.listdir(path)
    
    accession_number = []
    series_description = []
    
    for study in studies:
        #print("On Study: ", study)
        series_path = os.path.join(path,study)
        
        series = os.listdir(series_path)
                
        #Get the first dicom file in a series
        first_file = series[0]
        
        if ".dcm" in first_file:
            #print(first_file)
            scan = pydicom.read_file(os.path.join(series_path,first_file))
            
            if scan.AccessionNumber not in accession_number:
                accession_number.append(scan.AccessionNumber)
            if scan.SeriesDescription not in series_description:
                accession_number.append(scan.SeriesDescription)
        
    
    return accession_number, series_description
    

if __name__=="__main__":
    path = "/media/ubuntu/cryptscratch/cspine_data/batch1/Final/"
    
    acNum,serDesc = get_acNum_serDesc(path)
    
    print("Accession Numbers: ", acNum)
    print("Series Descriptions: ", serDesc)
