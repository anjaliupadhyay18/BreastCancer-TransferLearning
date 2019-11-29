import os
import shutil
import fnmatch


import modules

def gen_find(filepat,top):
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist,filepat):
            yield os.path.join(path,name)
#Attempt to create new directory for all images within subdirectory
try:
    os.mkdir('Collapsed')
    print('Directory Collapsed Created')
except FileExistsError:
    print('Directory Exists, Skipping')
#Set source location to point into extracted BreaKhis directory; 
#example:
src = '/home/jyue/Documents/BreaKHis_v1/' 
dst = 'Collapsed' 

print('Loading images into collapsed/ subdirectory')
filesToCopy = gen_find("*.png",src)
for name in filesToCopy:
    shutil.copy(name, dst)
print('Changing python work directory to collapsed/')
os.chdir('Collapsed')
samp_no = input('How many samples do you need from each class? ')
modules.sampling(int(samp_no))

