

import os
import requests
import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
refresh_token = 'PASTE YOUR REFRESH TOKEN HERE'
DOMAIN = 'https://mysleepscoring-api.cerebramedical.com/'

def getAccessToken():
    '''
    returns the active bearer token for the user
    '''
    # old account 
    data = {"refresh_token" : ''}
    
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
    '/media/erikjan/Expansion/OSD/Datasets/raw'
    #files = glob('/media/erikjan/Expansion/OSD/Datasets/raw/edfs/sub*/*/eeg/*.edf')
    # for i,file in tqdm(enumerate(files)):
    
    # fn = file.split('/')[-1].split('task')[0][:-1]
    ################
    # STEP 1-4
    ################
    #use refresh token to get access token
    token = getAccessToken()

    #get a list of files for the current user
    calculated = glob('/media/erikjan/Expansion/OSD/Datasets/raw/predictions/*')
    calculated = [x.split('/')[-1] for x in calculated]

    all_files = getFiles(token)
    for i in tqdm(np.arange(len(all_files))):
        
        
        file = all_files[-i]
        #use refresh token to get access token
        try: 
            token = getAccessToken()
        
            # scoringrun_id_val = default_scoring_run_id
            out_dir = '/media/erikjan/Expansion/OSD/Datasets/raw/predictions/'
            # #what file types you want to download
            download_types = ['detailed_orp_report'] #['Autoscoring Events','Report Data']

            scoring_run_info = getScoringRun(file['scoringruns'][0]['id'], token)

            if scoring_run_info['status'] == 'Reporting Complete':
                for file in scoring_run_info['files']:
                    file_id = file['id']
                    f_type = file['type']
                    file_name = file['study']['description'] #this should be your folder name

                    out_file_folder = os.path.join(out_dir, file_name)
                    if not os.path.exists(out_file_folder):
                        os.mkdir(out_file_folder)

                    if file['type'] in ['detailed_orp_report'] and file['type'] in download_types:
                        
                        out_f = os.path.join(out_file_folder, f'{file_name}_{f_type}.json')
                        if not os.path.exists(out_f):
                            print(f'\tfrom {file_name} downloading {f_type}')
                            my_json = downloadAutoScoringJson(file_id, token)
                            with open(out_f, 'w') as out_file:
                                    json.dump(my_json, out_file)
            else:
                pass

            #print('done')
        except:
            pass

