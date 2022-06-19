import os
import sys
import errno
import requests
import subprocess
import shutil
from IPython.display import HTML, display

FOLDERS = {
    0: ['prediction_models','plots','explore','dataset'],
    1: ['prediction_models','plots','explore','dataset'],
    2: ['prediction_models','plots','explore','dataset','dataset','dataset'],
}
FILENAMES = {
    0: ['genericRegressionClassification.py','stage1.py','analytics.py','uci/airfoil_noise/airfoil_self_noise.dat'],
    1: ['genericRegressionClassification.py','stage1.py','analytics.py','power_plant/pp_data.csv'],
    2: ['genericRegressionClassification.py','stage1.py','analytics.py','adult_census/adult.data','adult_census/adult.names','adult_census/adult.test'],
}

def download_to_colab(chapter, branch='master'):  

    base_url = 'https://raw.githubusercontent.com/manojmanivannan/machine-learning-with-PyTorch/{}/'.format(branch)

    folders = FOLDERS[chapter]
    filenames = FILENAMES[chapter]
    for folder, filename in zip(folders, filenames):
        if len(folder):
            try:
                os.mkdir(folder)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        if len(filename):
            path = os.path.join(folder, filename)
            try:
                os.makedirs(os.path.dirname(path))
            except FileExistsError:
                pass
            url = '{}{}'.format(base_url, path)
            r = requests.get(url, allow_redirects=True)
            open(path, 'wb').write(r.content)

    try:
        os.mkdir('runs')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
try:
    import google.colab
    IS_COLAB = True
except ModuleNotFoundError:
    IS_COLAB = False

def import_from_github(dataset=0, branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(dataset, branch)
        print('Finished!')
