#encoding:utf8
'''
Keras implementation of Bi-directional Spatio-Temporal Dependence and Users’Dynamic
Preferences Model( Bi-STDDP) based on tensorflow backend.

@author:
xidongbo17s@ict.ac.cn
'''
from keras.layers import Dense, Embedding, merge, Input,Reshape,Activation,Subtract,Multiply,Lambda,Dropout
from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.externals import joblib
from sklearn import preprocessing as process
import numpy as np
import os
import tensorflow as tf
import argparse
import keras.backend.tensorflow_backend as k
from itertools import izip
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpu_options = tf.GPUOptions(allow_growth=True)
sess=tf.InteractiveSession(
        config=tf.ConfigProto(
            gpu_options=gpu_options))
k.set_session(sess)

def parse_args():
    parser = argparse.ArgumentParser(description="Run STDDP.")
    parser.add_argument('--embedded_dim', type=int, default=64,
                        help='Embedding dim.')
    parser.add_argument('--hidden_unit', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--length', type=int, default=1,
                        help='Length of POI sequence.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--nb_epoch', type=int, default=50,
                        help='Number of epoch.')
    parser.add_argument('--earlystop', type=int, default=1,
                        help='Earlystop to avoid overfitting.')
    parser.add_argument('--model_name', type=str, default='STDDP.model',
                        help='Name of best model to save.')
    parser.add_argument('--dataset', type=str, default='NYC',
                        help='Which dataset to use.')
    parser.add_argument('--basepath', type=str, default="data/foursquare/",
                        help='Dataset path.')
    return parser.parse_args()

class STDDP(object):
    def __init__(self,embedded_dim,hidden_unit,length,dropout,batch_size,lr,nb_epoch,earlystop,model_name,dataset,basepath):
        self.embedded_dim=embedded_dim
        self.hidden_unit=hidden_unit
        self.length=length
        self.dropout=dropout
        self.batch_size=batch_size
        self.lr=lr
        self.nb_epoch=nb_epoch
        self.earlystop=earlystop
        self.model_name=model_name
        self.dataset=dataset
        self.basepath=basepath
        self.para = joblib.load(os.path.join(self.basepath, "{}.para".format(self.dataset)))
        self.poi_index = joblib.load(os.path.join(self.basepath, "poi_index_{}.pkl".format(self.dataset)))
        self.poi_loc = joblib.load(os.path.join(self.basepath, 'poi_loc.{}'.format(self.dataset)))
        self.maxlen = self.para['maxlen']
        self.nb_poi = self.para['nb_poi']
        self.nb_train = self.para['nb_train']
        self.nb_dev = self.para['nb_dev']
        self.nb_test = self.para['nb_test']
        self.nb_user = self.para['nb_user']
        print('maxlen:' + str(self.maxlen))
        # nb: Number
        print('nb_poi:' + str(self.nb_poi))
        print('nb_train:' + str(self.nb_train))
        print('nb_dev:' + str(self.nb_dev))
        print('nb_test:' + str(self.nb_test))
        print('nb_user:' + str(self.nb_user))
        self._init_model()

    def _init_model(self):
        # Model
        ###################user embedding#############
        user = Input(shape=(1,), dtype='int32')
        user_vec = Embedding(input_dim=self.nb_user + 1,
                             output_dim=self.embedded_dim)(user)
        user_vec = Reshape((self.embedded_dim,))(user_vec)
        ###############################################

        ###################poi embedding#############
        x1 = Input(shape=(self.length,), dtype='int32')  # prior POI sequence
        x2 = Input(shape=(self.length,), dtype='int32')  # next POI sequence
        # share embedding vector
        embedding = Embedding(
            input_dim=self.nb_poi + 1,
            output_dim=self.embedded_dim)
        prior_pois = embedding(x1)
        next_pois = embedding(x2)
        # None,length,dim->None,length*dim
        prior_pois = Reshape((self.length * self.embedded_dim,))(prior_pois)
        next_pois = Reshape((self.length * self.embedded_dim,))(next_pois)
        ############################################

        ###################Bi-STDDP#############
        x1_t = Input(shape=(1,), dtype='float32')  # prior POI's time(seconds)
        x2_t = Input(shape=(1,), dtype='float32')  # next POI's time(seconds)
        x1_d = Input(shape=(self.nb_poi,), dtype='float32')  # prior POI's distance vec of all candidate POIs
        x2_d = Input(shape=(self.nb_poi,), dtype='float32')  # next POI's distance vec of all candidate POIs
        y_t1 = Input(shape=(7,), dtype='float32')  # target time pattern
        y_t2 = Input(shape=(1,), dtype='float32')  # target time

        # --Bi-STD: Bi-directional Spatio-Temporal Dependence
        sub1 = Subtract()([y_t2, x1_t])
        sub2 = Subtract()([x2_t, y_t2])
        sub1 = Dense(self.nb_poi, activation='tanh')(Lambda(lambda x: x / 3600.)(sub1))
        sub2 = Dense(self.nb_poi, activation='tanh')(Lambda(lambda x: x / 3600.)(sub2))
        x1_dis = Multiply()([sub1, x1_d])  # STD_{t-1}
        x2_dis = Multiply()([sub2, x2_d])  # STD_{t+1}
        # --DP: Dynamic Preference
        wt = Dense(self.hidden_unit, activation='tanh')(y_t1)
        wp1 = Dense(self.hidden_unit, activation='tanh')(prior_pois)
        wp2 = Dense(self.hidden_unit, activation='tanh')(next_pois)
        user_vec = Dense(self.hidden_unit, activation='tanh')(user_vec)
        dynamic_preference = merge([wp1, wp2, user_vec, wt], mode='sum', output_shape=(self.hidden_unit,))

        output = Dense(self.nb_poi)(dynamic_preference)
        output=Dropout(self.dropout)(output)
        output = merge([output, x1_dis, x2_dis], mode='sum')
        output = Activation('softmax')(output)

        self.model = Model(inputs=[x1, x2, x1_t, x2_t, x1_d, x2_d, y_t1, y_t2, user], outputs=output)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=adam(
                lr=self.lr),
            metrics=[
                self.acc_top1,
                self.acc_top5,
                self.acc_top10])
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=self.earlystop)
        self.checkpoint = ModelCheckpoint(self.model_name, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    def fit(self):
        self. model.fit_generator(
            generator=self.generate_data_from_file(type='train'),
            steps_per_epoch=np.ceil(self.nb_train/self.batch_size).astype(int), epochs=self.nb_epoch, verbose=1,
            callbacks=[self.early_stopping, self.checkpoint],
            validation_data=self.generate_data_from_file(type='dev'),
            validation_steps=np.ceil(self.nb_dev/self.batch_size).astype(int), max_queue_size=20, workers=1)

    def test(self):
        self.model.load_weights(self.model_name, {'acc_top1': self.acc_top1, 'acc_top5': self.acc_top5, 'acc_top10': self.acc_top10})
        print(self.model.evaluate_generator(generator=self.generate_data_from_file(type='test'),
                                       steps=np.ceil(self.nb_test/self.batch_size).astype(int), max_queue_size=20, workers=1))

        count = 0
        MAP = 0
        step = 0
        steps = np.ceil(self.nb_test/self.batch_size).astype(int)
        for x, y_true in self.generate_data_from_file(type='test'):
            step += 1
            y_priord = self.model.predict(x)
            for i in range(len(y_true)):
                count += 1
                ziped = zip(y_true[i], y_priord[i])
                ziped.sort(key=lambda x: x[1], reverse=True)
                p = [t[0] for t in ziped]
                rank = np.argmax(p) + 1
                MAP += 1.0 / rank
            if step >= steps:
                break
        print('test samples:%d' % count)
        print('MAP:%.4f' % (MAP / count))

    def acc_topk(self,y_true, y_pred, k):
        topk = tf.nn.top_k(y_pred, k).indices
        y = tf.argmax(y_true, axis=-1)
        y = tf.reshape(y, (-1, 1))
        y = tf.cast(y, dtype=tf.int32)
        acc = tf.equal(y, topk)
        return tf.reduce_mean(tf.cast(acc, dtype=tf.float32)) * acc.shape[1].value

    def acc_top1(self,y_true, y_priord):
        return self.acc_topk(y_true, y_priord, k=1)

    def acc_top5(self,y_true, y_priord):
        return self.acc_topk(y_true, y_priord, k=5)

    def acc_top10(self,y_true, y_priord):
        return self.acc_topk(y_true, y_priord, k=10)



    def generate_data_from_file(self,type='train'):
        if type == 'train':
            x_path = 'train.{}'.format(self.dataset)
            x_st_path = 'train_st.{}'.format(self.dataset)
            y_path = 'train_label_poi.{}'.format(self.dataset)
        elif type == 'dev':
            x_path = 'dev.{}'.format(self.dataset)
            x_st_path = 'dev_st.{}'.format(self.dataset)
            y_path = 'dev_label_poi.{}'.format(self.dataset)
        elif type == 'test':
            x_path = 'test.{}'.format(self.dataset)
            x_st_path = 'test_st.{}'.format(self.dataset)
            y_path = 'test_label_poi.{}'.format(self.dataset)
        else:
            print('data type error')
            exit()
        flag = True
        # POI sequence information and user. Split by space.
        # Each line: forward_POI_sequence(self.maxlen) + backward_POI_sequence(self.maxlen)+user
        x_path = os.path.join(self.basepath, x_path)

        # Neighbor POI's temporal and spatio information. Split by space.
        # Each line: prior POI' visit time+next POI' visit time
        x_st_path = os.path.join(self.basepath, x_st_path)

        # the missing POI(label) information. Split by space.
        # Each line: missing_POI_name+target_time_pattern(7-dim)+target_time(seconds)
        y_path = os.path.join(self.basepath, y_path)
        while True:
            with open(x_path) as f1, open(x_st_path) as f2, open(y_path) as f3:
                count = 0
                for x_line, x_st_line, y_line in izip(f1, f2, f3):
                    y_line=y_line.split()
                    label_poi = y_line[0] # missing_POI
                    label_t = np.array(y_line[1:8], dtype=np.float32) # target_time_pattern(7-dim)
                    label_second = float(y_line[8]) # target_time(seconds)
                    onehot_poi = np.zeros(len(self.poi_index) + 1, dtype=np.int)
                    #if self.poi_index.has_key(label_poi):
                    pindex = self.poi_index.get(label_poi,0)
                    onehot_poi[pindex] = 1
                    if flag:
                        # x1 and x2 means forward and backward, s means spatio, t means temporal
                        x1, x2, x1_t, x2_t, x1_s, x2_s, y, y_t1, y_t2, user = [], [], [], [], [], [], [], [], [], []
                        #loc1_batch,loc2_batch=[],[] 
                    flag = False
                    count += 1
                    x_line = x_line.split()
                    x1.append(x_line[self.maxlen - self.length:self.maxlen]) # the forward POI sequence
                    x2.append(x_line[self.maxlen:self.maxlen + self.length]) # the backward POI sequence
                    user.append([x_line[-1]])

                    x_st_line = x_st_line.split()
                    x1_t.append([float(x_st_line[0])]) # the prior POI's visit time
                    x2_t.append([float(x_st_line[1])]) # the next POI's visit time
                    ###################################################
                    x1_s.append(self.poi_loc[int(x1[-1][-1])])# prior POI's space vector
                    x2_s.append(self.poi_loc[int(x2[-1][0])])# next POI's space vector
                    y.append(onehot_poi)
                    y_t1.append(label_t)
                    y_t2.append([label_second])
                    if count >= self.batch_size:
                        count = 0
                        flag = True
                        x1 = np.array(x1, dtype=np.int)
                        x2 = np.array(x2, dtype=np.int)
                        x1_t = np.array(x1_t, dtype=np.float32)
                        x2_t = np.array(x2_t, dtype=np.float32)
                        x1_s = np.array(x1_s, dtype=np.float32)
                        x2_s = np.array(x2_s, dtype=np.float32)
                        y_t1 = np.array(y_t1, dtype=np.float32)
                        y_t2 = np.array(y_t2, dtype=np.float32)
                        user = np.array(user, dtype=np.int)
                        y = np.array(y, dtype=np.int)
                        yield ([x1, x2, x1_t, x2_t, x1_s, x2_s, y_t1, y_t2, user], y)
                if not flag:
                    flag = True
                    x1 = np.array(x1, dtype=np.int)
                    x2 = np.array(x2, dtype=np.int)
                    x1_t = np.array(x1_t, dtype=np.float32)
                    x2_t = np.array(x2_t, dtype=np.float32)
                    x1_s = np.array(x1_s, dtype=np.float32)
                    x2_s = np.array(x2_s, dtype=np.float32)
                    y_t1 = np.array(y_t1, dtype=np.float32)
                    y_t2 = np.array(y_t2, dtype=np.float32)
                    user = np.array(user, dtype=np.int)
                    y = np.array(y, dtype=np.int)
                    yield ([x1, x2, x1_t, x2_t, x1_s, x2_s, y_t1, y_t2, user], y)

if __name__ == '__main__':
    args = parse_args()
    stddp=STDDP(embedded_dim=args.embedded_dim,hidden_unit=args.hidden_unit,length=args.length,batch_size=args.batch_size,
                lr=args.lr,nb_epoch=args.nb_epoch,earlystop=args.earlystop,model_name=args.model_name,dataset=args.dataset,
                basepath=args.basepath)
    stddp.fit()
    stddp.test()
