

import os
import requests
import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import mne

refresh_token = 'PASTE YOUR REFRESH TOKEN HERE'
DOMAIN = 'https://mysleepscoring-api.cerebramedical.com/'

def getAccessToken():
    '''
    returns the active bearer token for the user
    '''
    # old account 
    data = {"refresh_token" : '...'}
    
    url = f"{DOMAIN}/api/token"

    r = requests.post(url=url, json=data)
    r.raise_for_status()
    result = r.json()['access_token']
    return result

def getFiles(bearer_token):
    '''
    gets files for a user
    '''
    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = f"{DOMAIN}/api/file"

    all_results = []

    skip = 0
    limit = 5000

    while True:
        params = {"limit": limit, "skip": skip}

        r = requests.get(url=url, headers=headers, params=params)

        r.raise_for_status()
        result = r.json()
        all_results.extend(result)

        if len(result) < limit:
            break

        skip += limit

    return all_results

def createStudy(data, bearer_token):
    '''
    creates a study object
    '''
    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = f"{DOMAIN}/api/study/"

    r = requests.post(url=url, headers=headers, json=data)
    r.raise_for_status()
    result = r.json()
    return result

def createFile(data, bearer_token):
    '''
    creates a file object
    type = ['Raw EDF','RPSGT Edits']
    '''
    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = f"{DOMAIN}/api/file/"

    r = requests.post(url=url, headers=headers, json=data)
    r.raise_for_status()
    result = r.json()
    return result

def createChannelMapping(data, bearer_token):
    '''
    creates a channel mapping object
    '''
    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = f"{DOMAIN}/api/channelmapping/"

    r = requests.post(url=url, headers=headers, json=data)
    r.raise_for_status()
    result = r.json()
    return result

def getChannelMappings(bearer_token):
    '''
    creates a channel mapping object
    '''
    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = f"{DOMAIN}/api/channelmapping"

    r = requests.get(url=url, headers=headers)
    r.raise_for_status()
    result = r.json()
    return result


def patchFileAndChannelMapping(data,file_id, bearer_token):
    '''
    creates the association between a file and a channel mapping object
    '''
    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = f"{DOMAIN}/api/file/{file_id}"

    r = requests.patch(url=url, headers=headers, json=data)
    r.raise_for_status()
    result = r.json()
    return result

def getFile(file_id, bearer_token):
    '''
    get a file object, used to get the scoring run id
    '''
    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = f"{DOMAIN}/api/file/{file_id}"

    r = requests.get(url=url, headers=headers)
    r.raise_for_status()
    result = r.json()
    return result

def patchScoringRun(data,scoringrun_id, bearer_token):
    '''
    creates the association between a file and a channel mapping object
    '''
    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = f"{DOMAIN}/api/scoringrun/{scoringrun_id}"

    r = requests.patch(url=url, headers=headers, json=data)
    r.raise_for_status()
    result = r.json()
    return result

def uploadEDF(edf_file, presigned_url, bearer_token):
  '''
  upload an edf file to the URL
  '''
  headers = {"Authorization": f"Bearer {bearer_token}"}
  files = {'file': open(edf_file, 'rb')}
  r = requests.post(url=presigned_url, files=files)
  print(f"upload file to gcs status_code:{r.status_code}")
  return r.status_code

def downloadAutoScoringJson(file_id, bearer_token):
  '''
  get the json results for MY Sleep Scoring from Cerebra
  '''
  headers = {"Authorization": f"Bearer {bearer_token}"}
  url = f"{DOMAIN}/api/file/{file_id}/download"

  r = requests.get(url=url, headers=headers)
  r.raise_for_status()
  data = r.json()
  return data

def getScoringRun(scoringrun_id, bearer_token):
  '''
  get a file object, used to get the scoring run id
  '''
  headers = {"Authorization": f"Bearer {bearer_token}"}
  url = f"{DOMAIN}/api/scoringrun/{scoringrun_id}"

  r = requests.get(url=url, headers=headers)
  r.raise_for_status()
  result = r.json()
  return result


def graph_architecture(stats, png=None):
    '''
    creates a simple bar plot showing ORP architecture
    '''
    y_vals = []
    x_order = ['000_025','025_050','050_075','075_100','100_125','125_150','150_175','175_200','200_225','225_250']
    for x_val in x_order:
      if x_val in stats["orp"]["summary"]["decile"]["total_study_percent"].keys():
          y_vals.append(stats["orp"]["summary"]["decile"]["total_study_percent"][x_val])
      else:
          y_vals.append(0)

    bad_data = stats["orp"]["summary"]["decile"]["total_study_percent"]["bad"]

    plt.bar(x=range(len(x_order)), height=y_vals, label='ORP', color='#3AA0F5')
    plt.bar(x=10, height=bad_data, label='bad data', color='#D3D3D3')
    plt.ylabel('% of sleep study')
    plt.title('ORP Architecture')
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10], labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'n/a'])
    plt.legend()
    plt.tight_layout()
    if png:
        plt.savefig(png)
        plt.clf()
    else:
        plt.show()


def graph_orp(orp1, orp2, png=None):
    '''
    creates a simple graph showing an ORP time series
    '''
    plt.rcParams["figure.figsize"] = (8,3)
    plt.plot(orp1, label='EEG1')
    plt.plot(orp2, label='EEG2')
    plt.xlabel('Epoch')
    plt.ylabel('ORP')
    plt.title('30-second ORP values')
    plt.ylim([0,2.5])
    plt.legend()
    plt.tight_layout()
    if png:
        plt.savefig(png)
        plt.clf()
    else:
        plt.show()


def compute_orp_30s(x):
    '''
    computes 30-second average ORP values, skipping invalid epochs labelled as -1
    '''
    set_lengh = 10
    averages = []
    size_x = len(x)

    for i in range(0, len(x), set_lengh):
        subset = x[i:min(i+set_lengh, size_x)]
        all_valid = [value for value in subset if value != -1]

        if all_valid:
            average = sum(all_valid) / len(all_valid)
        else:
            average = None

        averages.append(average)

    return averages

def process_orp(events, report):
    '''
    read and graph 30-second ORP and ORP architecture
    '''
    event_info = json.load(open(events))
    report_info = json.load(open(report))

    eeg1_orp_3sec = event_info['EEG1ORPs']
    eeg2_orp_3sec = event_info['EEG2ORPs']

    eeg1_orp = compute_orp_30s(eeg1_orp_3sec)
    eeg2_orp = compute_orp_30s(eeg2_orp_3sec)

    graph_orp(eeg1_orp, eeg2_orp)
    graph_architecture(report_info)


from glob import glob

if __name__ == '__main__':
    WORKING_DIRECTORY = '/user/your/path/to/OSD_repo'
    files = glob(f'WORKING_DIRECTORY/Datasets/DATA_OSD/raw/edfs/sub*/*/eeg/*.edf')
    include_chan = ['C3-M2','C4-M1','E1-M2','E2-M1','CHIN1-CHIN2','Chin-Chin2']

    for i,file in enumerate(tqdm(files)):
        try: 
            
            fn = file.split('/')[-1].split('task')[0][:-1]
            ################
            # STEP 1-4
            ################
            #use refresh token to get access token
            

            # #get a list of files for the current user
            if i ==0:
                token = getAccessToken()
                all_files = getFiles(token)
                n = [i['name'] for i in all_files]
            if f'{fn}.edf' in n:
                continue
            
            if f'{fn}.edf' not in n:
                continue

            token = getAccessToken()

            #create a study object, retain id for future API calls
            body = {
            "description": fn
            }
            study_details = createStudy(body, token)
            #print(study_details)

            #create a file object to get a pre-signed URL to upload to
            #note - the file extension in body["name"] is a new requirement (2024-06-06)
            body = {
            "name": f'{fn}.edf',
            "description": "Level II home PSG",
            "study_id": study_details['id'],
            "type": "Raw EDF"
            }
            file_details = createFile(body, token)


            ################
            # STEP 5
            ################
            #define channel mappings for your EDF configuration and scoring settings
            
            
            channels = mne.io.read_raw_edf(file).ch_names
            channels = [c.upper() for c in channels]
            try:
                body = {
                "name": "Sample PSG Channels",
                "eeg_left": {"channel_name": "C3-M2","channel_index": channels.index("C3-M2")},
                "eeg_right": {"channel_name": "C4-M1","channel_index": channels.index("C4-M1")},
                "eye_left": {"channel_name": "E1-M2","channel_index": channels.index("E1-M2")},
                "eye_right": {"channel_name": "E2-M1","channel_index": channels.index("E2-M1")},
                "chin": {"channel_name": "Chin1-Chin2","channel_index": channels.index("CHIN1-CHIN2")}
                }
            except:
                body = {
                "name": "Sample PSG Channels",
                "eeg_left": {"channel_name": "C3-M2","channel_index": channels.index("C3-M2")},
                "eeg_right": {"channel_name": "C4-M1","channel_index": channels.index("C4-M1")}
                }
            #create object
            url_val = createChannelMapping(body, token)

            #get all channel mappings available to user
            all_cms = getChannelMappings(token)

            ##########################
            #step 7
            ##########################

            body = {"channelmapping_id": url_val['id']}
            file_id = file_details['id']

            url_val = patchFileAndChannelMapping(body, file_id, token)
            #time.sleep(15)

            upload_url = url_val['upload_url']

            ################
            # step ...
            ################

            #from Part 1
            file_id_val = file_id

            #get file
            file_info = getFile(file_id_val, token)

            #first item in the list is the default scoring run
            default_scoring_run_id = file_info['scoringruns'][0]['id']

            #patch scoring run
            body = {
            'lights_off_epcoh' : 1,
            'lights_on_epcoh' : -1,
            'collection_system': 'Alice',
            'hypopnea_criteria':'B',
            'prefilter': 0}

            url_val = patchScoringRun(body, default_scoring_run_id, token)
            #time.sleep(15)

            ##################
            # upload data
            ##################

            file_to_upload = file
            
            try:
                up_val = uploadEDF(file_to_upload, upload_url, token)
            except:
                chan_to_include = [c for c in channels if c in include_chan]
                data = mne.io.read_raw_edf(file,include=chan_to_include)
                mne.export.export_raw(f'/media/erikjan/Expansion/OSD/Datasets/raw/edfs/temp_{i}.edf',data)
                up_val = uploadEDF(f'/media/erikjan/Expansion/OSD/Datasets/raw/edfs/temp_{i}.edf', upload_url, token)
                os.remove(f'/media/erikjan/Expansion/OSD/Datasets/raw/edfs/temp_{i}.edf')


            print('done')
        except:
            pass

