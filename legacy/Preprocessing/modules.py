import os
import shutil
import glob
import random
import fnmatch
random.seed(0)


def counter():
    counts = {
    'F' : 0,
    'PT' : 0,
    'A' : 0,
    'TA' : 0,
    'DC' : 0,
    'LC' : 0, 
    'MC' : 0,
    'PC' : 0
    }
    for name in glob.glob('SOB_*.png'):
        if fnmatch.fnmatch(name, 'SOB_?_F*.png'):
            counts['F'] += 1
        if fnmatch.fnmatch(name, 'SOB_?_PT*.png'):
            counts['PT'] += 1
        if fnmatch.fnmatch(name, 'SOB_?_A*.png'):
            counts['A'] += 1
        if fnmatch.fnmatch(name, 'SOB_?_TA*.png'):
            counts['TA'] += 1
        if fnmatch.fnmatch(name, 'SOB_?_DC*.png'):
            counts['DC'] += 1
        if fnmatch.fnmatch(name, 'SOB_?_LC*.png'):
            counts['LC'] += 1
        if fnmatch.fnmatch(name, 'SOB_?_MC*.png'):
            counts['MC'] += 1
        if fnmatch.fnmatch(name, 'SOB_?_PC*.png'):
            counts['PC'] += 1
    return counts
def batch_copy(filelist, dst):
    for filename in filelist:
        base, extension = os.path.splitext(filename)
        copied_name = os.path.join(dst, filename)
        if not os.path.exists(copied_name):
            shutil.copy(filename, dst)
        else:
            idx = 1
            while True:
                new_name = os.path.join(dst, base + "_" + str(idx) + extension)
                if not os.path.exists(new_name):
                    shutil.copy(filename, new_name)
                    break
                idx += 1
                    
def give_subclass_randomsamples(subclass, n_sample):
    if subclass == 'F':
        files = glob.glob('SOB_?_F*.png')
    if subclass == 'PT':
        files = glob.glob('SOB_?_PT*.png')
    if subclass == 'A':
        files = glob.glob('SOB_?_A*.png')
    if subclass == 'TA':
        files = glob.glob('SOB_?_TA*.png')
    if subclass == 'DC':
        files = glob.glob('SOB_?_DC*.png')
    if subclass == 'LC':
        files = glob.glob('SOB_?_LC*.png')
    if subclass == 'MC':
        files = glob.glob('SOB_?_MC*.png')
    if subclass == 'PC':
        files = glob.glob('SOB_?_PC*.png')
    filenames = random.choices(files,  k = n_sample)
    return filenames
def duplicate(file, dst):
    shutil.copy(file, dst+'/'+file.split('.')[0]+'_1'+'.png')

def create_stochastic_duplicates(subclass, no_samples, dst):
    tmpfn = give_subclass_randomsamples(subclass, no_samples)
    batch_copy(tmpfn, dst)
def sub_sample(subclass, no_samples, dst):
    tmpfn = give_subclass_randomsamples(subclass, no_samples)
    batch_copy(tmpfn, dst)
def sampling(no_sub_sam):
    try:
        os.mkdir('samples')
        print('Directory Created')
    except FileExistsError:
        print('Directory samples/ Exists, Skipping')
    dst = 'samples'
    counts = counter()
    for i in ['F', 'PT', 'A', 'TA', 'DC', 'LC', 'MC', 'PC']:
        try:
            os.mkdir('samples/'+i)
            print('Creating Sub Class Directory '+i+'/' )
        except FileExistsError:
            print('Sub Class Directory '+i+'/Exsists, Skipping')
        if no_sub_sam > counts[i]:
            diff = no_sub_sam - counts[i]
            sub_sample(i, counts[i], dst+'/'+i)
            create_stochastic_duplicates(i, diff, dst+'/'+i)
        elif no_sub_sam < counts[i]:
            sub_sample(i, no_sub_sam, dst+'/'+i)

   
