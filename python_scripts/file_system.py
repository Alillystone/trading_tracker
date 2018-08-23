import os
import shutil

csv_folder = 'csvs/'
ticker_data_folder = csv_folder + 'tickers/'

def directory_exists(path):
    return os.path.isdir(path)

def remove_directory(path):
    shutil.rmtree(path)

def create_directory(path):
    os.makedirs(path)

def file_exists(path):
    return os.path.isfile(path)

def delete_file(path):
    os.remove(path)