import cv2
import itertools, os, time
import numpy as np
#from Model import get_Model
#from parameter import letters
import argparse
from SamplePreprocessor import preprocess
from keras import backend as bknd
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import SGD
from keras import optimizers
from keras.utils import *
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from config2 import *

print('asdfasdf')
def decode_label(out):
    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(characters):
            outstr += characters[i]
    return outstr


#parser = argparse.ArgumentParser()
#parser.add_argument("-w", "--weight", help="weight file directory",
#                    type=str, default="Final_weight.hdf5")
#parser.add_argument("-t", "--test_img", help="Test image directory",
#                    type=str, default="./DB/test/")
#args = parser.parse_args()

def predict(test_dir):
    # Get CRNN model
    # build model
    inputShape = Input((width, height, 1))  # base on Tensorflow backend
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputShape)
    batchnorm_1 = BatchNormalization()(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_1)
    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_2)
    batchnorm_3 = BatchNormalization()(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(batchnorm_3)

    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_3)
    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_4)
    batchnorm_5 = BatchNormalization()(conv_5)
    pool_5 = MaxPooling2D(pool_size=(2, 2))(batchnorm_5)

    conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_5)
    conv_7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_6)
    batchnorm_7 = BatchNormalization()(conv_7)

    bn_shape = batchnorm_7.get_shape()  # (?, {dimension}50, {dimension}12, {dimension}256)

    '''----------------------STN-------------------------'''
    # you can run the model without this STN part by commenting out the STN lines then connecting batchnorm_7 to x_reshape,
    # which may bring you a higher accuracy
    #stn_input_shape = batchnorm_7.get_shape()
    #loc_input_shape = (stn_input_shape[1].value, stn_input_shape[2].value, stn_input_shape[3].value)
    #stn = SpatialTransformer(localization_net=loc_net(loc_input_shape),
    #                         output_size=(loc_input_shape[0], loc_input_shape[1]))(batchnorm_7)
    '''----------------------STN-------------------------'''

    print(bn_shape)  # (?, 50, 7, 512)

    # reshape to (batch_size, width, height*dim)
    # x_reshape = Reshape(target_shape=(int(bn_shape[1]), int(bn_shape[2] * bn_shape[3])))(stn_7)
    x_reshape = Reshape(target_shape=(int(bn_shape[1]), int(bn_shape[2] * bn_shape[3])))(batchnorm_7)

    fc_1 = Dense(128, activation='relu')(x_reshape)  # (?, 50, 128)

    print(x_reshape.get_shape())  # (?, 50, 3584)
    print(fc_1.get_shape())  # (?, 50, 128)

    rnn_1 = LSTM(128, kernel_initializer="he_normal", return_sequences=True)(fc_1)
    rnn_1b = LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(fc_1)
    rnn1_merged = add([rnn_1, rnn_1b])

    rnn_2 = LSTM(128, kernel_initializer="he_normal", return_sequences=True)(rnn1_merged)
    rnn_2b = LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(rnn1_merged)
    rnn2_merged = concatenate([rnn_2, rnn_2b])

    drop_1 = Dropout(0.25)(rnn2_merged)

    prediction = Dense(label_classes, kernel_initializer='he_normal', activation='softmax')(drop_1)

    model = Model(inputs = [inputShape], outputs=prediction)

    try:
        model.load_weights(model_path)
        print("...Previous weight data...")
    except:
        raise Exception("No weight file!")


    #test_dir =args.test_img
    test_dir = test_dir

    #test_imgs = os.listdir(args.test_img)
    total = 0
    acc = 0
    letter_total = 0
    letter_acc = 0
    start = time.time()
    #for test_img in test_imgs:
    img = cv2.imread(test_dir, cv2.IMREAD_GRAYSCALE)
    print("_______Img before preprocess: ",img.shape,"_____")
    out_img = preprocess(img, (width, height))
    out_img = np.expand_dims(out_img, -1)
    out_img = np.expand_dims(out_img, 0)
    print("_______Img after preprocess: ",out_img.shape,"_____")
    net_out_value = model.predict(out_img)

    #pred_texts = decode_label(net_out_value)

    #for i in range(min(len(pred_texts), len(test_img[0:-4]))):
    #    if pred_texts[i] == test_img[i]:
    #        letter_acc += 1
    #letter_total += max(len(pred_texts), len(test_img[0:-4]))

    #if pred_texts == test_img[0:-4]:
    #    acc += 1
    #total += 1
    #print('Predicted: %s  /  True: %s' % (label_to_hangul(pred_texts), label_to_hangul(test_img[0:-4])))
    
    # cv2.rectangle(img, (0,0), (150, 30), (0,0,0), -1)
    # cv2.putText(img, pred_texts, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

    #cv2.imshow("q", img)
    #if cv2.waitKey(0) == 27:
    #   break
    #cv2.destroyAllWindows()

    end = time.time()
    total_time = (end - start)
    #print("Time : ",total_time / total)
    #print("ACC : ", acc / total)
    #print("letter ACC : ", letter_acc / letter_total)
    return net_out_value

if __name__ == '__main__':
    x = predict(test_dir)
    print('_______After predict_______')
    f= open("guru99.txt","w+")
    np.set_printoptions(threshold='nan')
    #f.write(np.array2string(x))
    f.close()
    print(x.shape)
    predicted_txt = decode_label(x)
    print(predicted_txt)
    
