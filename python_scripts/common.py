import os
import shutil
import datetime

def ensure_directory_exists(directory):
    if (not os.path.exists(directory)):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)

def retrieve_current_date():
    return datetime.datetime.today()