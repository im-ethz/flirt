eda-explorer
============

Scripts to detect artifacts and in electrodermal activity (EDA) data. Note that these scripts are written for Python 2.7 and Python 3.7 


Version 1.0

Please also cite this project:
Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., & Picard, R. Automatic identification of artifacts in electrodermal activity data. In Engineering in Medicine and Biology Conference. 2015.


Required python packages can be found in requirements.txt

To run artifact detection from the command line:
==
python EDA-Artifact-Detection-Script.py

Currently there are only 2 classifiers to choose from: Binary or Multiclass

To run peak detection:
==
python EDA-Peak-Detection-Script.py

Descriptions of the algorithm settings can be found at http://eda-explorer.media.mit.edu/info/

To run accelerometer feature extraction:
==
python AccelerometerFeatureExtractionScript.py

This file works slightly differently than the others in that it gives summary information over periods of time.

Notes:
===

1. Currently, these files are written with the assumption that the sample rate is an integer power of 2. 

2. Please visit [eda-explorer.media.mit.edu](https://eda-explorer.media.mit.edu)
 to use the web-based version
