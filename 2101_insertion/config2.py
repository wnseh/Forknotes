import string

width = 200
height = 31
label_len = 16

#characters = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-'"?_"""
characters = string.printable
label_classes = len(characters)

# for batch_generator
lexicon_dic_path = 'mjsynth.tar/mnt/ramdisk/max/90kDICT32px/lexicon.txt'
#file_list = open('mjsynth.tar/mnt/ramdisk/max/90kDICT32px/annotation_train.txt', 'r')
#file_list_val = open('mjsynth.tar/mnt/ramdisk/max/90kDICT32px/annotation_val.txt', 'r')
img_folder = 'mjsynth.tar/mnt/ramdisk/max/90kDICT32px/90kDICT32px'
test_dir = 'mjsynth.tar/mnt/ramdisk/max/90kDICT32px/testImage/test.png'


# for CRNN_with_STN
learning_rate = 0.0001  # learning rate, 0.002 for default
weight_path = 'model/weights_best_STN.04-16.35.hdf5'
model_path = 'model/weights_for_predict_STN.hdf5'
cp_save_path = 'mjsynth.tar/mnt/ramdisk/max/weights_best_STN.{epoch:02d}-{loss:.2f}.hdf5' \
    # save checkpoint path
base_model_path = 'mjsynth.tar/mnt/ramdisk/max/weights_for_predict_STN.hdf5'  \
    # the model for predicting
tb_log_dir = 'mjsynth.tar/mnt/ramdisk/max/logs'  # TensorBoard save path, Optional
load_model_path = ''  \
    # if you want to train a new model, please set  load_model_path = ""

