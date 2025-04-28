# Ordinal Sleep Depth  

## Description  

### Background
Sleep is a critical physiological process that plays a vital role in maintaining overall health and well-being. Polysomnography (PSG), the gold standard for assessing sleep, captures various physiological signals including brain activity, eye movements, and muscle tone. Traditionally, sleep has been classified into distinct stages (NREM, REM), but this stage-based classification does not fully capture the continuous nature of sleep depth. This limitation has made it challenging to assess subtle variations in sleep depth, which could have implications for understanding sleep disorders and their relationship to health outcomes.

### Method
In response to this gap, we developed the **Ordinal Sleep Depth (OSD)** algorithm, which classifies 6-channel PSG data into a continuous measure of sleep depth with high temporal resolution. The algorithm is trained on 21,787 polysomnography recordings from 18,116 unique patients, using deep learning to extract temporal features that represent the depth of sleep across time. Unlike traditional sleep stage classification, which groups sleep into discrete stages, OSD provides a continuous ordinal scale of sleep depth that can be tracked throughout the night.

### Results
The OSD model was evaluated on a diverse set of PSG data and demonstrated superior sensitivity in detecting subtle variations in sleep depth compared to conventional methods. The model was shown to be highly correlated with clinical measures of sleep quality and arousal indices, and its continuous output allowed for a more nuanced understanding of sleep dynamics. OSD offers a higher temporal resolution and can capture dynamic changes in sleep depth that are often overlooked in traditional sleep stage classification.

### Conclusion
Ordinal Sleep Depth provides a novel, continuous measure of sleep depth derived from PSG data. This repository offers the full codebase to replicate the results from our study, including scripts for preprocessing data, training the model, and generating statistics and figures. The OSD model can be used to investigate sleep depth in clinical and research settings, providing valuable insights into sleep disorders and their relationship with health.

For related work on **Odds Ratio Product (ORP)**, please visit the [Cerebra Portal](https://docs.cerebraportal.com/) for additional resources. This repository enables users to either train the model from scratch, use pre-trained weights, or predict all files, while also providing the option to focus solely on reproducing the statistical analyses and figures from the paper.

By making the OSD model accessible and reproducible, we aim to facilitate its use in both clinical and research settings, helping to improve sleep diagnostics and treatment strategies in the future.


## About this repository
This repository provides the full codebase to reproduce the results from our paper (DOI: XXX). The repo focuses on the **reproducibility of the OSD model** and allows you to:  
- Train the model from scratch using the provided scripts.  
- Use pre-trained weights to replicate the results from the paper.  
- use the predicted files (OSD and ORP), and reproduce statistics and figures.  
For **ORP** reproduction, please visit [Cerebra Portal](https://docs.cerebraportal.com/) for additional resources.

## Future use of **OSD**
For the OSD-specific functionality, the script `RUN_OSD.py` is recommended, as it includes the scaling of the output. This is not included in `Step_3_predict_no_scaling.py`.

## Installation & Setup  
### Manual Installation  
1. Install Python (recommended version: `>=3.x`).  
2. Clone the repository:  
   ```bash  
   git clone https://github.com/bdsp-core/Ordinal-Sleep-Depth.git  
   cd Ordinal-Sleep-Depth
3. Install dependencies:
    pip install -r requirements.txt  
4. Download the dataset from [https://www.bdsp.io/](https://www.bdsp.io/) and store all .h5 files in the DATA_OSD/ folder.
5. Run the scripts in order:
    Step_0_pre_process_data.py  
    Step_1_train_model.py  
    Step_3_predict_no_scaling.py  
    Step_5_generate_results_file.py  
    Step_6_run_statistics.py  
    Step_7_generate_figures.py  
    Step_8_create_hypno_PSD_fig.py
    
    (optionally)  
    Optional_Step_4_1_get_ORP.py
    Optional_Step_4_2_retrieve_ORP.py
    Optional_Step_4_3_merge_OSD_ORP.py


## Technology Stack  
- **Python**  
- **TensorFlow**  

## Contributing  
Currently, contributions are not accepted.  

## License  
No license specified yet.  

## Authors  
**Erik-Jan Meulenbrugge, Msc, Haoqi Sun, PhD, Wolfgang Ganglberger, PhD, Samaneh Nasiri, PhD, Robert J. Thomas, MD, M. Brandon Westover, MD, PhD**




