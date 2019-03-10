import openpyxl
import os
import numpy as np
import argparse

def get_paths(is_fracture):
    #Get spreadsheet that contains paths to CSpine Accession Numbers
    book = openpyxl.load_workbook('CS_accession_number_paths.xlsx')
    sheet = book.active
    row_count = sheet.max_row
    
    if is_fracture:
        scan_file_name = "path_fractures.log"
    else:
        scan_file_name = "path_no_fractures.log"
        
    scans_log = open(scan_file_name,"w")
    
    
    for row in range(2,row_count+1):
        #Fractures
        if is_fracture:
            if sheet.cell(row,3).value == 1 or sheet.cell(row,4).value == 1:
                path_scan = sheet.cell(row,2).value
            
                if path_scan != None:   
                    scans_log.write(path_scan + "\n")
        #No Fractures
        else:
            if sheet.cell(row,3).value != 1 and sheet.cell(row,4).value != 1:
                path_scan = sheet.cell(row,2).value
            
            if path_scan != None:
                scans_log.write(path_scan + "\n")
    
    scans_log.close()
     
    return scan_file_name


if __name__ == '__main__':

    #True if we want to get paths with only fractures
    #False if we want to get paths without any fractures  
    is_fracture = False

    #Call Function
    file_name = get_paths(is_fracture)

    #Sort File According to Paths (for ease of reading)
    paths = open(file_name,"r")
    path_lines = paths.readlines()
    path_lines.sort()
    paths.close()

    paths = open(file_name,"w")

    for line in path_lines:
         paths.write(line)

    paths.close()

    
    


