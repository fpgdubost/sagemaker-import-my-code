import os
from shutil import copy2
import sys
from ipdb import set_trace as bp


def createExpFolderandCodeList(save_path,files=[]):
    #result folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    code_folder_path = os.path.join(save_path, 'code')
    if not os.path.exists(code_folder_path):
        os.makedirs(code_folder_path)
    #save code files
    for file_name in os.listdir('.') + files:
        if not os.path.isdir(file_name):
            copy2('./%s' % file_name, os.path.join(save_path, 'code', file_name))