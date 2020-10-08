import os
import tarfile
import shutil

classes_info = {}
with open('classes.txt', 'r') as f:
    temp = f.read().splitlines()
    name_labels = [' '.join(k) for k in [i.split()[1:] for i in temp]]
    num_labels = [int(i.split()[0]) - 1 for i in temp]
    for i in range(len(num_labels)):
        classes_info[name_labels[i]] = num_labels[i]
    del temp, name_labels, num_labels

tarpath = os.path.join(os.getcwd(), 'ip102_v1.1-002.tar')

traintxt = os.path.join(os.getcwd(), 'train.txt')
testtxt = os.path.join(os.getcwd(), 'test.txt')
validtxt = os.path.join(os.getcwd(), 'val.txt')

tb_name = {'train' : traintxt,
           'test' : testtxt,
           'valid' : validtxt}

head = 'ip102_v1.1/images/'

tar = tarfile.open(tarpath)
for setname, txtname in tb_name.items():
    with open(txtname, 'r') as f:
        temp = f.read().splitlines()
        img_names = [head + i.split()[0] for i in temp]
        labels = [int(i.split()[1]) for i in temp]
        del temp

    d_tarinfo = {}

    for label, name in zip(labels, img_names):
        if str(label) not in d_tarinfo:
            d_tarinfo[str(label)] = []
            d_tarinfo[str(label)].append(tar.getmember(name))
        else:
            d_tarinfo[str(label)].append(tar.getmember(name))

    path_folder = os.path.join(os.getcwd(), setname)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    ccount = 0
    for key, value in classes_info.items():

        ccount += 1
        path_class = os.path.join(path_folder, key)
        if not os.path.exists(path_class):
            os.makedirs(path_class)
        tar.extractall(path_class, members= d_tarinfo[str(value)])

        destination = path_class
        fsource = os.path.join(destination, 'ip102_v1.1', 'images')
        for name in os.listdir(fsource):
            shutil.move(os.path.join(fsource, name), destination)
        delete = os.path.join(path_class, 'ip102_v1.1')
        if os.path.exists(delete):
            shutil.rmtree(os.path.join(destination, 'ip102_v1.1'))
            print('deleted')
        print(key, 'done!', ccount)
tar.close()

for sets in ['train', 'test', 'valid']:

    count = 0
    path_folder = os.path.join(os.getcwd(), sets)

    for key, value in classes_info.items():
        destination = os.path.join(path_folder, key)
        temp = len(os.listdir(destination))
        count += temp

    print('total', sets, count)