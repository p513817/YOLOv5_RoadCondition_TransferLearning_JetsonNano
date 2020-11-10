import shutil
import os
import numpy as np

"""  取得客製化資料  """

def make_custom_data():

    ############## Check dir ##############
    dataset_dir ='data'
    label_dir = 'label'
    dataset = os.listdir(dataset_dir)
    label = os.listdir(label_dir)

    images = []
    labels = []

    custom_dir = 'custom'
    data_root = os.path.join(custom_dir, 'images')
    label_root = os.path.join(custom_dir, 'labels')

    data_train_dir = os.path.join(data_root, 'train')
    data_val_dir = os.path.join(data_root, 'val')
    label_train_dir = os.path.join(label_root, 'train')
    label_val_dir = os.path.join(label_root, 'val')

    dir_list = [data_train_dir, data_val_dir, label_train_dir, label_val_dir]

    for dir in dir_list:
        if os.path.exists(dir)==False:
            print(f"Create dir : {dir}")
            os.makedirs(dir) 

    ############## Get val data ##############

    #打亂數據
    def shuffle(data):
        
        arr = np.array(data)
        np.random.shuffle(arr)
        return arr.tolist()

    #拼接檔名
    def get_path(src, name, ftype):
        
        return f'{src}/{name}.{ftype}'

    val_num = 5
    pure_name = [ i.split('.')[0] for i in dataset ]
    val_data = shuffle(pure_name)[:val_num]

    print('Total Data length: ', len(pure_name))
    print('Validation Data: ', val_data)

    for d in val_data:

        src_data = get_path(dataset_dir, d, 'jpg')
        src_label = get_path(label_dir, d, 'txt')
        trg_data = get_path(data_val_dir, d, 'jpg')
        trg_label = get_path(label_val_dir, d, 'txt')

        shutil.copy(src_data, trg_data)
        shutil.copy(src_label, trg_label)

        pure_name.remove(d)

    ############## Split data ##############

    print("="*50)
    print('New Data length: ', len(pure_name))

    for d in pure_name:

        src_data = get_path(dataset_dir, d, 'jpg')
        src_label = get_path(label_dir, d, 'txt')
        trg_data = get_path(data_train_dir, d, 'jpg')
        trg_label = get_path(label_train_dir, d, 'txt')

        shutil.copy(src_data, trg_data)
        shutil.copy(src_label, trg_label) 

    print("Finish！")

if __name__ == "__main__":
    
    make_custom_data()