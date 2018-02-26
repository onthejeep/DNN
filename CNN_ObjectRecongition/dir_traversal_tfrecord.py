
import os;
import tensorflow as tf;

# define a function to list tfread files
def list_tfrecord_file(file_list):
    tfrecord_list = [];
    for file in tfrecord_list:
        current_file_abs_path = os.path.abspath(file);
        if current_file_abs_path.endswith('.tfdata'):
            tfrecord_list.append(current_file_abs_path);
            print('Found {} successfully '.format(file));
        else:
            pass;
    return tfrecord_list;

# traverse current directory
def tfrecord_auto_traversal(folderName = 'data/image'):
    current_folder_filename_list = os.listdir(folderName);
    if current_folder_filename_list != None:
        print('{} files were found under current folder'.format(len(current_folder_filename_list)));
        print('Please note that only files end with *.tfrecord will be loaded');
        tfrecord_list = list_tfrecord_file(current_folder_filename_list);
        if len(tfrecord_list) == 0:
            print('cannot find any tfrecord files, please check the path');
    return tfrecord_list;


def Find_TFdata(folderName = 'data/image'):
    TFList = [];
    for file in os.listdir(folderName):
        if file.endswith('.tfdata'):
            TFList.append(os.path.abspath(folderName + '/' + file));
    return TFList;
