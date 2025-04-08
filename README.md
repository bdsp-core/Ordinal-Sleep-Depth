# Ordinal Sleep Depth  

## Description  
Ordinal Sleep Depth is an algorithm that classifies 6-channel PSG (polysomnography) data into a continuous sleep depth measure with high temporal resolution. This repository provides the full codebase to reproduce the project as described in our paper (DOI: XXX).  

Users can:  
- Train the model from scratch using the provided scripts.  
- Use pre-trained weights to replicate the results from the paper.  
- Use a pre-built Docker image from BDSP.IO for easy deployment.  

## Installation & Setup  

### Option 1: Using the Pre-Built Docker Image (Recommended)  
1. Install [Docker](https://docs.docker.com/get-docker/).  
2. Download the Docker image from BDSP.IO and load it:  
   ```bash
   docker load < ordinal_sleep_depth_image.tar >
   ```  
3. Run the container:  
   ```bash
   docker run --rm -v $(pwd)/data:/app/data ordinal-sleep-depth
   ```  


### Option 2: Manual Installation  
1. Install Python (recommended version: `>=3.x`).  
2. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/ordinal-sleep-depth.git
   cd ordinal-sleep-depth
   ```  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
4. Download the dataset from [BDSP.IO](https://www.bdsp.io) and store all `.h5` files in the `DATA_OSD/` folder.  
5. Run the scripts in order:  
   ```bash
   python Step_0_pre_process_data.py
   python Step_1_train_model.py
   python Step_2_predict_So.py
   python Step_4_1_get_ORP.py
   python Step_4_2_retrieve_ORP.py
   python Step_5_merge_OSD_ORP.py
   python Step_6_1_generate_results_file_TRAIN.py
   python Step_6_2_generate_results_file.py
   python Step_7_0_STATISTICS_TRAIN_OSD.py
   python Step_7_1_STATISTICS_TEST_OSD.py
   python Step_8_1_generate_figures_train.py
   python Step_8_generate_figures.py
   python Step_9_create_hypno_PSD_fig.py
   ```  

## Features  
✅ Classifies PSG signals into ordinal sleep depth  
✅ High temporal resolution for sleep depth estimation  
✅ Fully reproducible results from the associated publication  
✅ Supports Docker for easy deployment  

## Technology Stack  
- **Python**  
- **Tensorflow**  
- **Keras**  
- **Docker**  

## Contributing  
Currently, contributions are not accepted.  

## License  
No license specified yet.  

## Authors  
**Erik-Jan Meulenbrugge** & **Michael B. Westover MD PhD.**  