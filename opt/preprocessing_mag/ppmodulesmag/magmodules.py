import os
import shutil
import glob
import random
import fnmatch
random.seed(0)


def counter():

    counts = {'40':
        {
        'F' : 0,
        'PT' : 0,
        'A' : 0,
        'TA' : 0,
        'DC' : 0,
        'LC' : 0,
        'MC' : 0,
        'PC' : 0
        },
            '100':
        {
        'F' : 0,
        'PT' : 0,
        'A' : 0,
        'TA' : 0,
        'DC' : 0,
        'LC' : 0,
        'MC' : 0,
        'PC' : 0
        },
            '200':
        {
        'F' : 0,
        'PT' : 0,
        'A' : 0,
        'TA' : 0,
        'DC' : 0,
        'LC' : 0,
        'MC' : 0,
        'PC' : 0
        },
            '400':
        {
        'F' : 0,
        'PT' : 0,
        'A' : 0,
        'TA' : 0,
        'DC' : 0,
        'LC' : 0,
        'MC' : 0,
        'PC' : 0
        }}
    for mag in ['40','100','200','400']:
        print(mag)
        for name in glob.glob('SOB_*.png'):

            if fnmatch.fnmatch(name, 'SOB_?_F*-'+mag+'-*.png'):
                counts[mag]['F'] += 1
            if fnmatch.fnmatch(name, 'SOB_?_PT*-'+mag+'-*.png'):
                counts[mag]['PT'] += 1
            if fnmatch.fnmatch(name, 'SOB_?_A*-'+mag+'-*.png'):
                counts[mag]['A'] += 1
            if fnmatch.fnmatch(name, 'SOB_?_TA*-'+mag+'-*.png'):
                counts[mag]['TA'] += 1
            if fnmatch.fnmatch(name, 'SOB_?_DC*-'+mag+'-*.png'):
                counts[mag]['DC'] += 1
            if fnmatch.fnmatch(name, 'SOB_?_LC*-'+mag+'-*.png'):
                counts[mag]['LC'] += 1
            if fnmatch.fnmatch(name, 'SOB_?_MC*-'+mag+'-*.png'):
                counts[mag]['MC'] += 1
            if fnmatch.fnmatch(name, 'SOB_?_PC*-'+mag+'-*.png'):
                counts[mag]['PC'] += 1


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

def give_subclass_randomsamples_wr(subclass, n_sample,mag):
    if subclass == 'F':
        files = glob.glob('SOB_?_F*-'+mag+'-*.png')
    if subclass == 'PT':
        files = glob.glob('SOB_?_PT*-'+mag+'-*.png')
    if subclass == 'A':
        files = glob.glob('SOB_?_A*-'+mag+'-*.png')
    if subclass == 'TA':
        files = glob.glob('SOB_?_TA*-'+mag+'-*.png')
    if subclass == 'DC':
        files = glob.glob('SOB_?_DC*-'+mag+'-*.png')
    if subclass == 'LC':
        files = glob.glob('SOB_?_LC*-'+mag+'-*.png')
    if subclass == 'MC':
        files = glob.glob('SOB_?_MC*-'+mag+'-*.png')
    if subclass == 'PC':
        files = glob.glob('SOB_?_PC*-'+mag+'-*.png')
    filenames = random.choices(files,  k = n_sample)
    return filenames
def give_subclass_randomsamples(subclass, n_sample,mag):
    if subclass == 'F':
        files = glob.glob('SOB_?_F*-'+mag+'-*.png')
    if subclass == 'PT':
        files = glob.glob('SOB_?_PT*-'+mag+'-*.png')
    if subclass == 'A':
        files = glob.glob('SOB_?_A*-'+mag+'-*.png')
    if subclass == 'TA':
        files = glob.glob('SOB_?_TA*-'+mag+'-*.png')
    if subclass == 'DC':
        files = glob.glob('SOB_?_DC*-'+mag+'-*.png')
    if subclass == 'LC':
        files = glob.glob('SOB_?_LC*-'+mag+'-*.png')
    if subclass == 'MC':
        files = glob.glob('SOB_?_MC*-'+mag+'-*.png')
    if subclass == 'PC':
        files = glob.glob('SOB_?_PC*-'+mag+'-*.png')
    filenames = random.sample(files,  n_sample)
    return filenames
def duplicate(file, dst):
    shutil.copy(file, dst+'/'+file.split('.')[0]+'_1'+'.png')

def create_stochastic_duplicates(subclass, no_samples, dst,mag):
    tmpfn = give_subclass_randomsamples(subclass, no_samples,mag)
    batch_copy(tmpfn, dst)
def sub_sample(subclass, no_samples, dst,mag):
    tmpfn = give_subclass_randomsamples(subclass, no_samples,mag)
    batch_copy(tmpfn, dst)
def sampling(no_sub_sam):
    try:
        os.mkdir('samples')
        print('Directory Created')
    except FileExistsError:
        print('Directory samples/ Exists, Skipping')
    dst = 'samples'
    counts = counter()
    for mag in ['40','100','200','400']:
        dst = 'samples/'+mag
        print(dst)
        try:
            os.mkdir('samples/'+mag)
            print('Creating Sub Magnification Directory '+mag+'/' )
        except FileExistsError:
            print('Sub Magnification Directory '+mag+'/Exsists, Skipping')

        for i in ['F', 'PT', 'A', 'TA', 'DC', 'LC', 'MC', 'PC']:
            try:

                os.mkdir('samples/'+mag+'/'+i)
                print('Creating Sub Class Directory '+i+'/' )
            except FileExistsError:
                print('Sub Class Directory '+i+'/Exsists, Skipping')
            if no_sub_sam >= counts[mag][i]:
                diff = no_sub_sam - counts[mag][i]
                sub_sample(i, counts[mag][i], dst+'/'+i,mag)
                create_stochastic_duplicates(i, diff, dst+'/'+i,mag)
            elif no_sub_sam < counts[mag][i]:
                sub_sample(i, no_sub_sam, dst+'/'+i,mag)
