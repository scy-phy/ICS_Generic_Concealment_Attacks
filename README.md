Assessing Model-free Anomaly Detection in Industrial Control Systems Against Generic Concealment Attacks
=======

## In proceedings of the the Annual Computer Security Applications Conference (ACSAC)

When using this code from this repository please cite our work as follows:
```
BIBTEX entry to be added
``` 
 
## Implementation of spoofing framework, re-implementation of detection mechanisms from prior work and our defense
  
### Description
  
 This repository is organized as follows:

  * `Spoofing Framework` contains the spoofing framework presented in subsection 4.7 of the paper, and implements the attacks detailed in subsections 4.5
  * `AR` contains our re implementation of AR+CUSUM detection presented in section 5.
  * `PASAD` contains our adaptation of the PASAD detector to work with BATADAL dataset as presented in section 5. Our adaptation is based on the official code by the authors of the PASAD paper [Github](https://github.com/mikeliturbe/pasad) 


### Requirements

To execute and evaluate the framework `Python 3`, `Keras`, `Pandas`, `Numpy`, `Jupyter` and `Matlab` are required. Installation through `conda` is suggested.

### Usage

#### Spoofing Framework folder:

##### spoofing_framework.py

Creates the datasets containing the generic concealment attacks and stores them in csv format.

This version is tuned to work with the BATADAL dataset.
Parameters at the beginning of the main can be changed to adapt to other datasets.

Usage:

`python spoofing_framework.py`

#### AR Folder:

##### AR_detection.m

MATLAB function that provides AR detection with CUSUM (as described in section 5)
This function is used by the next two scripts

##### find_cusum_params.m

CUSUM tuning script that performs grid search (as described in appendix A).

Usage:

execute it in Matlab

##### test_detection.m

Test the generic concealment attacks against the AR detector (as in section 6.2.1).

The output of the script produces Table 4

Usage:

execute it in Matlab

#### PASAD Folder:

##### pasad_BATADAL.m

Test the generic concealment attacks against the PASAD detector (as in section 6.2.2).

The output of the script produces Table 5

Usage:

execute it in Matlab

