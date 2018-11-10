import openpyxl
import os
import numpy as np

book = openpyxl.load_workbook('CS_accession_number_paths_2018_1025.xlsx')
sheet = book.active
row_count = sheet.max_row
#print(row_count)

paths = []

for row in range(2,row_count+1):
    if sheet.cell(row,3).value == 1 or sheet.cell(row,4).value == 1:
        path_frac = sheet.cell(row,2).value

        if path_frac != None:
            paths.append(path_frac)
            print(path_frac)

#print(paths[1:20])
