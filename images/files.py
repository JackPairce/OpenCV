import os
import shutil
from typing import List, TypeVar

target = [3206, 3549, 3342, 3283]

def get_files():
    n = len(target)
    files = os.listdir()
    folders = [folder for folder in files if os.path.isdir(folder)]
    for folder in folders:
        targets = os.listdir(folder)
        for image in targets:
            if int(image.split('.')[0]) in target:
                print(folder+'/'+ image)
                shutil.copy(folder+'/'+ image, 'target/'+image)
                n-=1
                if not n: return

if __name__ == '__main__':
    get_files()


