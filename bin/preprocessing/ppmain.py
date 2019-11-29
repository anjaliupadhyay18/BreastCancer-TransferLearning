import os
import shutil
import fnmatch
from bin.preprocessing.ppmodules import modules


def preprocess(src = '/home/jyue/Documents/BreaKHis_v1/'):
    os.chdir('dataset')
    #Attempt to create new directory for all images within subdirectory
    try:
        os.mkdir('Collapsed')
        print('Directory Collapsed Created')
    except FileExistsError:
        print('Directory Exists, Skipping')
    #Set source location to point into extracted BreaKhis directory; 
    #example:
     
    dst = 'Collapsed' 

    print('Loading images into collapsed/ subdirectory')
    filesToCopy = modules.gen_find("*.png",src)
    for name in filesToCopy:
        shutil.copy(name, dst)
    print('Changing python work directory to collapsed/')
    os.chdir('Collapsed')
    samp_no = input('How many samples do you need from each class? ')
    modules.sampling(int(samp_no))
    os.chdir('..')
    os.chdir('..')
if __name__ == '__main__':
	preprocess()
