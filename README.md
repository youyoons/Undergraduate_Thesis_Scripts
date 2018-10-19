# Undergraduate_Thesis_Scripts
This is a collection of the scripts that I have used (that are not in other repositories) for my undergraduate EngSci thesis.

all_acnum_serdesc.py

Purpose: to get all the accession numbers in a batch using get_accession number(); to get some series descriptions in a btach using get_series_description()

Please run this script as:
python all_acnum_serdesc.py CRYPT BATCH | tee acnum_serdesc_BATCH.txt
where CRYPT is either cryptscratch or cryptscratch3, and batch can be batch1 to batch16. Please also replace BATCH in the tee command with the corresponding batch (i.e. batch1). 

Example: python all_acnum_serdesc.py cryptscratch batch1 | tee acnum_serdesc_batch1.txt

Note: I will be working on changing this script such that it will be able to save info as pickle objects.
