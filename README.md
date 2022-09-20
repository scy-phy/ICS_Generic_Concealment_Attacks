Assessing Model-free Anomaly Detection in Industrial Control Systems Against Generic Concealment Attacks
=======

## In proceedings of the the Annual Computer Security Applications Conference 2022 (ACSAC'2022)

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
  * `SFIG` folder contains our re-implementation of the SFIG detector presented in section 5 . For the CFP-Growth++  our script relies on v2.40 spmf library available at [http://www.philippe-fournier-viger.com/spmf/](http://www.philippe-fournier-viger.com/spmf/)
  * `Autoencoders` folder relies on anomaly detection [https://github.com/scy-phy/ICS-Evasion-Attacks](https://github.com/scy-phy/ICS-Evasion-Attacks) as described in section 5.
  * `Countermeasure` this contains the anomaly detection system presented in the appedix C

### Requirements

To execute and evaluate the framework `Python 3`, `Keras` (with tensorflow backend), `Pandas`, `Numpy`, `Jupyter`, `Java JDK`, and `Matlab` (with System Identification Toolbox) are required. Installation through `conda` is suggested.

```
pip install -r requirements.txt
```

### Virtual machine availability
For the purposes of artifact evalaution a virtual machine is available for the reviewers. Please contact us via HotCRP comments to access the VM. 


### Spoofing Framework folder:

#### spoofing_framework.py

Creates the datasets containing the generic concealment attacks and stores them in csv format.

This version is tuned to work with the BATADAL dataset.
Parameters at the beginning of the main can be changed to adapt to other datasets.

Usage:

```
python spoofing_framework.py 
```

### AR Folder:

#### AR_detection.m

MATLAB function that provides AR detection with CUSUM (as described in section 5)
This function is used by following two scripts

#### find_cusum_params.m

CUSUM tuning script that performs grid search (as described in appendix A).

Usage:
```
execute it in Matlab
```

#### test_detection.m

Test the generic concealment attacks against the AR detector (as in section 6.2.1).

The output of the script produces Table 4

Usage:
```
execute it in Matlab
```
### PASAD Folder:

#### pasad_BATADAL.m

Test the generic concealment attacks against the PASAD detector (as in section 6.2.2).

The output of the script produces Table 5

Usage:
```
execute it in Matlab
```
### SFIG Folder:

#### invariant_mining.py
**OPTIONAL**
This script is used to mine the invariant rules for the anomaly detector. 
**Note** to reproduce the detector used in the paper it needs to run for 12 hours, you can run it for 600 seconds to get an approximately good detector.
the model will be stred in the `model_folder` that is set in line 132 of the script
Usage:

```
python invariant_mining.py
```

#### anomaly_detection.py
This script performs anomaly detection. It is configured to reprodice the results in Section 6.2.3 (Table 6 and Figure 4).
Results will be stored in the `./model_folder/results.csv` (where model_folder is set at line 86 of the script).

(Expect it to take some hours (4/5 hours) to produce all the results).

Usage:

```
python anomaly_detection.py
```

### Autoencoders Folder:

This folder builds on prior work Autoencoder anomaly detection [https://github.com/scy-phy/ICS-Evasion-Attacks](https://github.com/scy-phy/ICS-Evasion-Attacks).
The scripts from the original artifact are modified to evalaute the generic concealment attacks as in Section 6.2.4, Figure 5 and Table 7

#### ./Evaluation/Evaluation_BATADAL.ipynb

execute the python notebook to reproduce results showed in Figure 5 and Table 7

### Countermeasure Folder:

this folder contains the countermeasure presented in the appedix

#### ensemble_BATADAL.m

Train and Test countermeasure on the original BATADAl data and on the generic concealment attacks (as in Appendix C, Table 10, Figures 8 and 9).

Usage:
```
execute it in Matlab
```
