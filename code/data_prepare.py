"""This file use for prepare the data for training"""

import os
import tarfile
import shutil

def process_IP102(root):
    necessaryFile = ['train.txt', 'test.txt', 'val.txt', 'classes.txt', 'ip102_v1.1-002.tar']
    checklist = [0 for _ in range(len(necessaryFile))]

    for filename in os.listdir(root):
        checklist[necessaryFile.index(filename)] = 1
    if not all(checklist):
        listMissing = []
        for i, chk in enumerate(checklist):
            if chk == 0:
                listMissing.append(necessaryFile[i])
        raise Exception('File(s) ' + ' '.join(listMissing) + ' missing in the root folder')

    classes_info = {}
    with open(os.path.join(root, 'classes.txt'), 'r') as f:
        temp = f.read().splitlines()
        name_labels = [' '.join(k) for k in [i.split()[1:] for i in temp]]
        num_labels = [int(i.split()[0]) - 1 for i in temp]
        for i in range(len(num_labels)):
            classes_info[name_labels[i]] = num_labels[i]
        del temp, name_labels, num_labels

    tarpath = os.path.join(root, 'ip102_v1.1-002.tar')

    traintxt = os.path.join(root, 'train.txt')
    testtxt = os.path.join(root, 'test.txt')
    validtxt = os.path.join(root, 'val.txt')

    tb_name = {'train' : traintxt,
            'test' : testtxt,
            'valid' : validtxt}

    head = 'ip102_v1.1/images/'

    root_path_folder = os.path.join(os.getcwd(), 'IP102')
    if not os.path.exists(root_path_folder):
        os.makedirs(root_path_folder)

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

        path_folder = os.path.join(root_path_folder, setname)
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

def process_D0(root, unzip_file= True):
    import zipfile
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import re

    list_label = []
    dataset = {}
    root_folder = os.path.join(os.getcwd(), 'unzip_D0')

    if unzip_file:
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        for name in os.listdir(os.path.join(root, 'D0')):
            with zipfile.ZipFile(os.path.join(root, 'D0', name), 'r') as zip_ref:
                zip_ref.extractall(root_folder)

    for name in os.listdir(root_folder):
        n_name = re.sub("[^a-z A-Z]+", "", name)
        list_label.append(n_name)
        dataset[name] = os.listdir(os.path.join(os.getcwd(), root_folder, name))

    encoder = LabelEncoder().fit(list_label)

    X = []
    y = []

    for name, list_image_name in dataset.items():
        for image_name in list_image_name:
            n_name = re.sub("[^a-z A-Z]+", "", name)
            a = os.path.join(os.getcwd(), root_folder, name, image_name)
            b = encoder.transform([n_name])[0]
            X.append(a)
            y.append(b)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42, shuffle= True, stratify= y)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size= 0.3, random_state= 42, shuffle= True, stratify= y_test)

    for setfolname, X_data, y_data in zip(['train', 'test', 'valid'], 
    [X_train, X_test, X_valid], [y_train, y_test, y_valid]):
        setfolder_path = os.path.join(root_folder, setfolname)
        if not os.path.exists(setfolder_path):
            os.makedirs(setfolder_path)
        for name in list_label:
            if not os.path.exists(os.path.join(setfolder_path, name)):
                os.makedirs(os.path.join(setfolder_path, name))

        for x, y in zip(X_data, y_data):
            clsname = encoder.inverse_transform([y])[0]
            shutil.move(x, os.path.join(setfolder_path, clsname.strip()))

    for sets in ['train', 'test', 'valid']:
        count = 0
        path_folder = os.path.join(root_folder, sets)

        for folder_name in os.listdir(path_folder):
            temp = len(os.listdir(os.path.join(path_folder, folder_name)))
            count += temp

        print('total', sets, count)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-data", "--dataset", required=True, help= "Choose dataset: IP102 or D0")
ap.add_argument("-root", "--rootdataset", required=True, help= "Choose dataset path")
args = vars(ap.parse_args())

def invalidError():
    raise Exception('Invalid dataset, type python data_prepare.py -help for detail')

if __name__ == "__main__":
    dataset_name = args['dataset']
    dataset_path = args['rootdataset']

    switcher = {'IP102' : process_IP102,
                'D0' : process_D0}

    switcher.get(dataset_name, invalidError)(dataset_path)