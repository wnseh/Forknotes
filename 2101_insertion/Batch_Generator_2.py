import cv2
import numpy as np
import linecache
from SamplePreprocessor import preprocess

from config2 import width, height, label_len, lexicon_dic_path, file_list, file_list_val, img_folder, \
    characters
    
# load train data
file_list_full = file_list.readlines()
file_list_len = len(file_list_full)


# load validation data
file_list_val_full = file_list_val.readlines()
file_list_val_len = len(file_list_val_full)

image_dir = 'data/words_train/'
wordfile = open(image_dir + "words.txt", 'r')
wordfile_list = wordfile.readlines()
for x in wordfile_list:
    if x[0] == "#" or not x:
        wordfile_list.remove(x)
wordfile_list_len = len(wordfile_list)
image_dir_val = 'data/words_val/'
wordfile_val = open(image_dir + "words.txt", 'r')
wordfile_list_val = wordfile.readlines()
for x in wordfile_list_val:
    if x[0] == "#" or not x:
        wordfile_list_val.remove(x)
wordfile_list_len_val = len(wordfile_list)





def img_gen(batch_size=50, input_shape=None):
    x = np.zeros((batch_size, width, height, 1), dtype=np.uint8)
    y = np.zeros((batch_size, label_len), dtype=np.uint8)
    # y = np.zeros((batch_size, ), dtype=np.uint8)

    while True:
        for ii in range(batch_size):
            while True:  # abandon the lexicon which is longer than 16 characters
                pick_index = np.random.randint(0, wordfile_list_len - 1)
                line = wordfile_list[pick_index]
                #e.g. 'a01-000u-00-00 ok 154 408 768 27 51 AT A'
                #index:      0         1  2   3   4  5  6  7  8
                line_separation = line.split(' ')

                #e.g. 'a01-000u-00-00'
                #index: 0   1    2  3
                file_path_separation = line_separation[0].split('-')
                try:
                    #e.g.       '/Letsgo/data'                   'a01'                                            'a01-000u'                         'a01-000u-00-00'
                    img_path = image_dir + file_path_separation[0] + '/' + file_path_separation[0]+'-'+file_path_separation[1]+'/'+line_separation[0]+'.png'
                except IndexError:
                    print(IndexError)
                    break
                #some text values are not just one word, so we have to add all values after index 8
                line_length = len(line_separation)
                lexicon = str(' '.join(line_separation[8:line_length])).rstrip() #rstrip to get rid of the newline
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # abandon the lexicon which is longer than 16 characters, because I set the label_len = 16, you can change it anyway.
                # some dataset images may be damaged during unzip
                if (img is not None) and len(lexicon) <= label_len:
                    img_size = img.shape  # (height, width, channels) 
                    if img_size[1] > 2 and img_size[0] > 2: 
                        break
            #if (img_size[1]/img_size[0]*1.0) < 6.4:
                #img_reshape = cv2.resize(img, (int(31.0/img_size[0]*img_size[1]), height))
                #mat_ori = np.zeros((height, width - int(31.0/img_size[0]*img_size[1]), 1), dtype=np.uint8)
                #out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
            #else:
                #out_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                #print("DEBUGGING:_______", out_img.shape)
                #out_img = np.asarray(out_img).transpose([1, 0, 2])

            out_img = preprocess(img, (width, height))
            out_img = np.expand_dims(out_img, -1)


            # due to the explanation of ctc_loss, try to not add "-" for blank
            while len(lexicon) < label_len:
                lexicon += "-"


            x[ii] = out_img
            y[ii] = [characters.find(c) for c in lexicon]
        yield [x, y, np.ones(batch_size) * int(input_shape[1] - 2), np.ones(batch_size) * label_len], y
                #[input, labels, input length, label length], output


def img_gen_val(batch_size=1000):
    x = np.zeros((batch_size, width, height, 1), dtype=np.uint8)
    # y = np.zeros((batch_size, label_len), dtype=np.uint8)
    y = []

    while True:
        for ii in range(batch_size):
            while True:  # abandon the lexicon which is longer than 16 characters
                pick_index = np.random.randint(0, wordfile_list_len_val - 1)
                line = wordfile_list_val[pick_index]
                #e.g. 'a01-000u-00-00 ok 154 408 768 27 51 AT A'
                #index:      0         1  2   3   4  5  6  7  8
                line_separation = line.split(' ')

                #e.g. 'a01-000u-00-00'
                #index: 0   1    2  3
                file_path_separation = line_separation[0].split('-')
                try:
                    #e.g.       '/Letsgo/data'                   'a01'                                            'a01-000u'                         'a01-000u-00-00'
                    img_path = image_dir + file_path_separation[0] + '/' + file_path_separation[0]+'-'+file_path_separation[1]+'/'+line_separation[0]+'.png'
                except IndexError:
                    print(IndexError)
                    break
                #some text values are not just one word, so we have to add all values after index 8
                line_length = len(line_separation)
                lexicon = str(' '.join(line_separation[8:line_length])).rstrip() #rstrip to get rid of the newline
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # abandon the lexicon which is longer than 16 characters, because I set the label_len = 16, you can change it anyway.
                # some dataset images may be damaged during unzip
                if (img is not None) and len(lexicon) <= label_len:
                    img_size = img.shape  # (height, width, channels) 
                    if img_size[1] > 2 and img_size[0] > 2: 
                        break #if its a working image, break the loop
                
            out_img = preprocess(img, (width, height))
            out_img = np.expand_dims(out_img, -1)

            #if (img_size[1]/img_size[0]*1.0) < 6.4:
                #img_reshape = cv2.resize(img, (int(31.0/img_size[0]*img_size[1]), height))
                #mat_ori = np.zeros((height, width - int(31.0/img_size[0]*img_size[1]), 1), dtype=np.uint8)
                #out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
            #else:
                #out_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                #out_img = np.asarray(out_img).transpose([1, 0, 2])


            x[ii] = out_img
            y.append(lexicon)
        yield x, y
