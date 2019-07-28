import os
import pandas as pd
import time
from sklearn.externals import joblib
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing as process
from scipy.spatial.distance import cdist
class DataHelp(object):
    def __init__(self):
        self.url = r"data/"
        self.basepath = os.path.join(self.url, "foursquare/")
        # user whose number of check-in less this will be removed
        self.min_poi_num = 10
        self.min_left_right_length = 4
        # each sample length,0 padding
        self.max_len = 10
        self.nb_train, self.nb_dev, self.nb_test = 0, 0, 0
        self.data_list = []
        self.data = []

    def preprocess_foursquare(self, data='NYC'):
        print ('preprocess {} data'.format(data))

        datapath = os.path.join(
            self.basepath, "dataset_TSMC2014_{}.txt".format(data))
        data_df = pd.read_csv(datapath, header=None, sep='\t', encoding='utf8')
        # 0:user_id, 1:poi_id, 2:class_id, 3:class_name, 4:lati, 5:long, 6:offset, 7:utc_time
        # NYC_dict.data:
        # format:map{user_id:map{'poi_id':[],'class_name':[],'loc':[(1,2)],'time':[]}}
        self.nb_user=len(data_df[0].unique())
        print('user:%d' % self.nb_user)
        print('poi:%d' % len(data_df[1].unique()))
        
        train_data_dict = dict()
        dev_data_dict = dict()
        test_data_dict = dict()
        print ('user:%d' % len(data_df[0].unique()))
        print ('poi:%d' % len(data_df[1].unique()))
        # transfer to timestamp
        data_df[7] = [time.mktime(time.strptime(
            data_df[7][index], '%a %b %d %X +0000 %Y')) + int(data_df[6][index]) * 60 for index in data_df[7].index]
        #sort by time
        data_df = data_df.sort_values(by=7,axis=0,ascending=True)
        # NYC:1-1083
        user_id_list = list(data_df[0])
        poi_id_list = list(data_df[1])
        lati_list = list(data_df[4])
        long_list = list(data_df[5])
        uct_time_list = list(data_df[7])
        nb_check_in = len(user_id_list)
        print ('total %d check_ins' % nb_check_in)
        # 8:1:1split
        dev_split = '2012-12-14 12:00:00'  # '2012-08-14 12:00:00'  #
        test_split = '2013-01-15 12:00:00'  # '2012-11-15 12:00:00'  #
        dev_split_timetamp = time.mktime(
            time.strptime(dev_split, "%Y-%m-%d %H:%M:%S"))
        test_split_timetamp = time.mktime(
            time.strptime(test_split, "%Y-%m-%d %H:%M:%S"))
        for i in range(nb_check_in):
            check_in_timetamp = uct_time_list[i]
            user_id = user_id_list[i]
            if check_in_timetamp < dev_split_timetamp:  # train data
                if user_id not in train_data_dict:
                    train_data_dict[user_id] = dict()
                    train_data_dict[user_id]['poi_id'] = [poi_id_list[i]]
                    train_data_dict[user_id]['loc'] = [(float(lati_list[i]), float(long_list[i]))]
                    train_data_dict[user_id]['time'] = [check_in_timetamp]
                else:
                    train_data_dict[user_id]['poi_id'].append(poi_id_list[i])
                    train_data_dict[user_id]['loc'].append(
                        (float(lati_list[i]), float(long_list[i])))
                    train_data_dict[user_id]['time'].append(check_in_timetamp)
            elif check_in_timetamp > test_split_timetamp:  # test data
                if user_id not in test_data_dict:
                    test_data_dict[user_id] = dict()
                    test_data_dict[user_id]['poi_id'] = [poi_id_list[i]]
                    test_data_dict[user_id]['loc'] = [(float(lati_list[i]), float(long_list[i]))]
                    test_data_dict[user_id]['time'] = [check_in_timetamp]
                else:
                    test_data_dict[user_id]['poi_id'].append(poi_id_list[i])
                    test_data_dict[user_id]['loc'].append(
                        (float(lati_list[i]), float(long_list[i])))
                    test_data_dict[user_id]['time'].append(check_in_timetamp)
            else:  # dev data
                if user_id not in dev_data_dict:
                    dev_data_dict[user_id] = dict()
                    dev_data_dict[user_id]['poi_id'] = [poi_id_list[i]]
                    dev_data_dict[user_id]['loc'] = [(float(lati_list[i]), float(long_list[i]))]
                    dev_data_dict[user_id]['time'] = [check_in_timetamp]
                else:
                    dev_data_dict[user_id]['poi_id'].append(poi_id_list[i])
                    dev_data_dict[user_id]['loc'].append(
                        (float(lati_list[i]), float(long_list[i])))
                    dev_data_dict[user_id]['time'].append(check_in_timetamp)

        # remove less self.min_poi_num sequence
        train_temp = list(train_data_dict.keys())
        dev_temp = list(dev_data_dict.keys())
        test_temp = list(test_data_dict.keys())
        print ('total train %d users' % len(train_data_dict))
        print ('total dev %d users' % len(dev_data_dict))
        print ('total test %d users' % len(test_data_dict))
        print (
            'user whose number of check-in  less %d will be removed' %
            self.min_poi_num)
        for key in train_temp:
            if len(train_data_dict[key]['poi_id']) < self.min_poi_num:
                train_data_dict.pop(key)
        for key in dev_temp:
            if len(dev_data_dict[key]['poi_id']) < self.min_poi_num:
                dev_data_dict.pop(key)
        for key in test_temp:
            if len(test_data_dict[key]['poi_id']) < self.min_poi_num:
                test_data_dict.pop(key) 
        
        print ('total train %d users' % len(train_data_dict))
        print ('total dev %d users' % len(dev_data_dict))
        print ('total test %d users' % len(test_data_dict))
        joblib.dump([train_data_dict, dev_data_dict, test_data_dict], os.path.join(
            self.basepath, "{}_dict.data".format(data)), compress=3)

    def load_preprocessed_foursquare(self, data='NYC'):
        print ('load {} data'.format(data))
        datapath = os.path.join(self.basepath, "{}_dict.data".format(data))
        # data
        # format:map{user_id:map{'poi_id':[],'loc':[(1,2)],'time':[]}}
        train_data_dict, dev_data_dict, test_data_dict = joblib.load(datapath)
        #train_data_list's each entry is a list:[poi_sequences_list,poi_sequences_loc_list,poi_sequences_time_list,user]
        train_data_list = []
        #key is user
        for key in train_data_dict:
            #python 3 not use str()
            train_data_list.append([train_data_dict[key]['poi_id'],train_data_dict[key]['loc'],train_data_dict[key]['time'],key])
        dev_data_list = []
        for key in dev_data_dict:
            dev_data_list.append([dev_data_dict[key]['poi_id'],dev_data_dict[key]['loc'],dev_data_dict[key]['time'],key])
        test_data_list = []
        for key in test_data_dict:
            test_data_list.append([test_data_dict[key]['poi_id'],test_data_dict[key]['loc'],test_data_dict[key]['time'],key])
        self.nb_train, self.nb_dev, self.nb_test = len(
            train_data_list), len(dev_data_list), len(test_data_list)
        print ('before sample:train:%d, dev:%d, test:%d' %
               (self.nb_train, self.nb_dev, self.nb_test))
        self.train_data_list = train_data_list
        self.dev_data_list = dev_data_list
        self.test_data_list = test_data_list

    def encode_padding(self, data='NYC'):
        self.load_preprocessed_foursquare(data=data)
        counts = []
        data_list1 = []
        data_list2 = []
        # list1+label+list2
        data_list_lmr = []
        labels_poi = []
        left_neighbor_narray =[]
        right_neighbor_narray =[]
        label_narray=[]

        # train_data_list's each entry is a list:
        # [poi_sequences_list,poi_sequences_loc_list,poi_sequences_time_list,user]
        users=[]
        train_user=set()
        for data_list in [
                self.train_data_list,
                self.dev_data_list,
                self.test_data_list]:
            count = 0
            for pois in data_list:
                #[poi_sequences_list,poi_sequences_loc_list,poi_sequences_time_list,user]
                sequence = pois[0]
                loc_sequence = pois[1]
                time_sequence = pois[2]
                user=pois[3]
                if data_list==self.train_data_list:
                    train_user.add(user)
                elif user not in train_user:
                    #not in the train user is oov,index is 0
                    user=0
                for label_rand_index in range(self.min_left_right_length,len(sequence)-self.min_left_right_length):
                    count+=1
                    #user for user_vec
                    users.append(user)
                    left_sequence = sequence[max(label_rand_index-self.max_len,0):label_rand_index]
                    right_sequence = sequence[label_rand_index+1:min(label_rand_index+self.max_len+1,len(sequence))]

                    # time : 1
                    left_neighbor_narray.append([time_sequence[label_rand_index - 1]],)
                    right_neighbor_narray.append([time_sequence[label_rand_index + 1]])
                    label_time=self.encode_time_loc(time_sequence[label_rand_index])
                    label_time.append(time_sequence[label_rand_index])
                    label_narray.append(label_time)

                    sequence_lmr = sequence[
                                   max(label_rand_index - self.max_len, 0):min(label_rand_index + self.max_len + 1,
                                                                               len(sequence))]
                    data_list1.append(str(' '.join(left_sequence)))
                    data_list2.append(str(' '.join(right_sequence)))
                    data_list_lmr.append(str(' '.join(sequence_lmr)))
                    labels_poi.append(sequence[label_rand_index])
            counts.append(count)
        left_neighbor_narray=np.array(left_neighbor_narray)
        right_neighbor_narray=np.array(right_neighbor_narray)
        label_narray=np.array(label_narray)

        self.nb_train, self.nb_dev, self.nb_test = counts
        print('after sample:train:%d, dev:%d, test:%d' %
              (self.nb_train, self.nb_dev, self.nb_test))
        # set nb_words will result in  random
        #for poi sequence, string split using ' '
        tokenizer = Tokenizer(
            nb_words=None,
            filters='',
            lower=False,
            split=' ',
            char_level=False)
        self.poi_data=data_list_lmr
        #only count in train,not in train is oov,index is 0
        tokenizer.fit_on_texts(self.poi_data[:self.nb_train])
        self.poi_index = tokenizer.word_index
        joblib.dump(self.poi_index, os.path.join(
            self.basepath, 'poi_index_{}.pkl'.format(data)), compress=3)
        print('Found %d unique poi in train.' % len(self.poi_index))
        left_sequences = tokenizer.texts_to_sequences(data_list1)
        right_sequences = tokenizer.texts_to_sequences(data_list2)

        index_poi = dict(zip(self.poi_index.values(), self.poi_index.keys()))
        joblib.dump(index_poi, os.path.join(
            self.basepath,'index_poi_{}.pkl'.format(data)), compress=3)
        train_dev_num = self.nb_train + self.nb_dev

        left_lens = sorted([len(sequence) for sequence in left_sequences])
        left_zero_count = 0
        for length in left_lens:
            if length == 0:
                left_zero_count += 1
        print('left_samples minlen:%d,maxlen:%d,avglen:%d,zero_len_num:%d' %
              (left_lens[0], left_lens[-1], np.mean(left_lens),left_zero_count))

        right_lens = sorted([len(sequence) for sequence in right_sequences])
        right_zero_count = 0
        for length in right_lens:
            if length == 0:
                right_zero_count += 1
        print('right_samples minlen:%d,maxlen:%d,avglen:%d,zero_len_num:%d' %
              (right_lens[0], right_lens[-1], np.mean(right_lens),right_zero_count))
        self.nb_poi = len(self.poi_index)+1
        params={'maxlen':self.max_len,'nb_poi':self.nb_poi,
                'nb_train':self.nb_train,'nb_dev':self.nb_dev,'nb_test':self.nb_test,'nb_user':self.nb_user}
        print("pad...")
        # default maxlen is the length of the longest sequence
        #len(left_sequences)<=maxlen
        #post padding!!!
        data1 = pad_sequences(left_sequences, maxlen=self.max_len,padding='pre')
        data2 = pad_sequences(right_sequences, maxlen=self.max_len,padding='post')

        print('Shape of all left_data tensor:', data1.shape)
        print('Shape of all right_data tensor:', data2.shape)
        print('Shape of all left_neighbor_data tensor:', left_neighbor_narray.shape)
        print('Shape of all right_neighbor_data tensor:', right_neighbor_narray.shape)
        print('Shape of all label_narray tensor:', label_narray.shape)

        le=process.LabelEncoder()
        users = np.array(users)
        users=le.fit_transform(users)
        users = np.reshape(users, (-1, 1))
        ######################shuffle train#########################
        train_users = users[:self.nb_train]

        # each line:poi_name,time_vec,loc_vec
        labels_poi = np.concatenate([np.reshape(np.array(labels_poi), (-1, 1)), label_narray], axis=-1)
        train_labels_poi = labels_poi[:self.nb_train]
        train_data_poi1 = data1[:self.nb_train]
        train_data_poi2 = data2[:self.nb_train]
        train_left_neighbor_narray=left_neighbor_narray[:self.nb_train]
        train_right_neighbor_narray = right_neighbor_narray[:self.nb_train]

        shuffle_index = range(self.nb_train)
        np.random.shuffle(shuffle_index)
        train_users = list(train_users[shuffle_index])

        train_labels_poi = train_labels_poi[shuffle_index]
        train_data_poi1 = train_data_poi1[shuffle_index]
        train_data_poi2 = train_data_poi2[shuffle_index]
        train_left_neighbor_narray=train_left_neighbor_narray[shuffle_index]
        train_right_neighbor_narray=train_right_neighbor_narray[shuffle_index]

        #labels_poi=np.concatenate([train_labels_poi,labels_poi[self.nb_train:]],axis=0)
        np.savetxt(os.path.join(self.basepath, "train_label_poi.{}".format(data)),train_labels_poi,fmt='%s')
        np.savetxt(os.path.join(self.basepath, "dev_label_poi.{}".format(data)),labels_poi[self.nb_train:train_dev_num],fmt='%s')
        np.savetxt(os.path.join(self.basepath, "test_label_poi.{}".format(data)),labels_poi[train_dev_num:],fmt='%s')
        ###############################################################
        np.savetxt(os.path.join(self.basepath, 'train.{}'.format(data)),
                   np.concatenate((train_data_poi1,train_data_poi2,train_users),
                                  axis=-1), fmt='%d')
        np.savetxt(os.path.join(self.basepath, 'train_st.{}'.format(data)),
                   np.concatenate((train_left_neighbor_narray, train_right_neighbor_narray),
                                  axis=-1), fmt='%.6f')
        np.savetxt(os.path.join(self.basepath, 'dev.{}'.format(data)),
                   np.concatenate((data1[self.nb_train:train_dev_num], data2[self.nb_train:train_dev_num],users[self.nb_train:train_dev_num]),
                                  axis=-1), fmt='%d')
        np.savetxt(os.path.join(self.basepath, 'dev_st.{}'.format(data)),
                   np.concatenate((left_neighbor_narray[self.nb_train:train_dev_num], right_neighbor_narray[self.nb_train:train_dev_num]),
                                  axis=-1), fmt='%.6f')
        np.savetxt(os.path.join(self.basepath, 'test.{}'.format(data)),
                   np.concatenate((data1[train_dev_num:], data2[train_dev_num:],users[train_dev_num:]),
                                  axis=-1), fmt='%d')
        np.savetxt(os.path.join(self.basepath, 'test_st.{}'.format(data)),
                   np.concatenate((left_neighbor_narray[train_dev_num:],
                                   right_neighbor_narray[train_dev_num:]),
                                  axis=-1), fmt='%.6f')
        joblib.dump(params, os.path.join(self.basepath, "{}.para".format(data)), compress=3)
        self.poi_loc(data)

    def poi_loc(self,data):
        # nb_poi*2
        p_loc = np.zeros((self.nb_poi, 2), dtype=np.float32)##
        print('p_loc matrix shape:', p_loc.shape)
        # only count data in train ,rather than dev and test data
        # index 0 is oov poi
        for user in self.train_data_list:
            # [poi_sequences_list,poi_sequences_loc_list,poi_sequences_time_list,user]
            pois = user[0]
            loc = user[1]
            for index in range(len(pois)):
                cindex = self.poi_index[pois[index]]
                if p_loc[cindex][0] == 0.:
                    p_loc[cindex] = loc[index]
                else:# average loc for mulit same POI
                    p_loc[cindex] = (p_loc[cindex]+loc[index])/2.
        p_loc[0]=np.mean(p_loc[1:],axis=0)
        # nb_poi*nb_poi
        loc_matrix=cdist(p_loc, p_loc, 'euclidean')
        loc_matrix = process.scale(loc_matrix, with_mean=False, axis=-1)  # with_mean=True
        np.savetxt(os.path.join(self.basepath, "poi_loc_{}.txt".format(data)),loc_matrix,fmt='%.6f')
        joblib.dump(loc_matrix, os.path.join(self.basepath, "poi_loc.{}".format(data)), compress=3)

    def encode_time_loc(self,poi_time):
        #int
        #encode weekend and weekday in one week and 5 time slot in one day
        #each poi_time to be a 2+5 dim vector
        onehot_time=[0]*7
        #timestamp to week and time
        time_str=time.strftime("%w:%H%M",time.localtime(poi_time))
        time_str=time_str.split(':')
        if int(time_str[0]) in [0,6]:
            week_index=0
        elif int(time_str[0]) in [1,2,3,4,5]:
            week_index = 1
        else:
            print('int(time_str[0])'+str(time_str[0]))
            exit()
        onehot_time[week_index]=1
        hour_minute=int(time_str[1])
        if 800<=hour_minute<1130:
            index=2
        elif 1130<=hour_minute<1400:
            index=3
        elif 1400<=hour_minute<1730:
            index=4
        elif 1730<=hour_minute<2200:
            index=5
        elif hour_minute>=2200 or hour_minute<800:
            index=6
        else:
            print('time error:'+str(hour_minute))
            exit()
        onehot_time[index]=1
        return onehot_time

if __name__ == '__main__':
    datahelp=DataHelp()
    datahelp.preprocess_foursquare('NYC')
    datahelp.encode_padding('NYC')
