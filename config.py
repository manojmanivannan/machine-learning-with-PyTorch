import os
import sys
import errno
import requests
import subprocess
import shutil
from IPython.display import HTML, display

FOLDERS = {
    0: ['prediction_models','plots','explore','dataset'],
}
FILENAMES = {
    0: ['genericRegressionClassification.py','stage1.py','analytics.py','uci/airfoil_noise/airfoil_self_noise.dat'],
}

def download_to_colab(chapter, branch='master'):  

    base_url = 'https://raw.githubusercontent.com/manojmanivannan/machine-learning-with-PyTorch/{}/'.format(branch)

    folders = FOLDERS[chapter]
    filenames = FILENAMES[chapter]
    for folder, filename in zip(folders, filenames):
        print('--------------------------')  
        print('Folder name:',folder,'File name:',filename)
        if len(folder):
            try:
                os.mkdir(folder)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        if len(filename):
            path = os.path.join(folder, filename)
            try:
                print('Parent directory:',os.path.dirname(path))
                os.makedirs(os.path.dirname(path))
            except FileExistsError:
                pass
            url = '{}{}'.format(base_url, path)
            print('download URL:',url)
            print('Save path:',path)
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

def config_chapter0(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(0, branch)
        print('Finished!')