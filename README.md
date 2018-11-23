# 3D Detection, Identification, and Segmentation of CSpine Vertebrae
This repository contains all scripts and programs required to successfully and robustly detect the C1 and C2 vertebrae of the (cervical) spine, then segment the detected vertebrae.

Detection Flow
1. It is important to acquire a list of directory locations for DICOM scans that pertain to my study. A filelist is obtained by running the get_paths.py script with the input being a spreadsheet with information on the DICOM scans. The spreadsheet contains the Accession Number (a primary key of the database), the directory path, and whether fractures exist in C1 and C2. 

2. This filelist is referred to in the anonymization script, where each study (single directory path in filelist) is first copied over to a local workspace, then processed by removing all patient and institution information that is private. The processed studies are organized in such a way that the Study ID is replaced by the Accession Number as the directory name, making it easier to relate back to the spreadsheet in #1.

3. The individual DICOM files (.dcm) are then turned into a 3D numpy array for analysis. This is done through the DicomSubsampling.py script, which reads in one series at a time and converts the intensity information for all slices (one DICOM file is referred to as a slice) into a 3D array. This resulting full-size 3D array representation of the DICOM scan is then downsampled to 4x by 4x for the frontal view (sagittal view) and 2x for the depth. After, this file is stored as a pkl file to be stored for later use.

4. In the same DicomSubsampling.py script, there is a VolumeOfInterest function that is used to extract a sample to be used in the final detection step. The boundaries of the C1, C2 vertebrae are manually given and the result of this extraction is saved as a pkl file.

5. Steps 1-4 allow all the information needed for detection to be obtained. The final step is to use the 3D GHT program to detect C1, C2 vertebrae on a new DICOM scan. This is done by running multi-sample detection, followed by max pooling in the accumulator matrix (Please see Ballard paper for more information). Afterwards, Gaussian blurring is performed on the accumulator matrix and non-maximal suppression (NMS) is performed to isolate only a few points of potential detection. Finally, cross-correlation is performed by sliding the reference image over the query image to get the optimal, final detection point.


Extra
all_acnum_serdesc.py

Purpose: to get all the accession numbers in a batch using get_accession number(); to get some series descriptions in a btach using get_series_description()

Please run this script as:
python all_acnum_serdesc.py CRYPT BATCH | tee acnum_serdesc_BATCH.txt
where CRYPT is either cryptscratch or cryptscratch3, and batch can be batch1 to batch16. Please also replace BATCH in the tee command with the corresponding batch (i.e. batch1). 

Example: python all_acnum_serdesc.py cryptscratch batch1 | tee acnum_serdesc_batch1.txt

Note: I will be working on changing this script such that it will be able to save info as pickle objects.
