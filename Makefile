SHELL = /bin/sh

get_paths:
	python3 Copy_Scans/get_paths.py

.PHONY : get_paths
