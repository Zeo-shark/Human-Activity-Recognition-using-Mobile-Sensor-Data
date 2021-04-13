import numpy as np # Multidimensional array Computation
import os #For Reading Necessary files
import pandas as pd # Reading files and storing to Dataframes

Data_Path="..\\data\\dataset"
Train_Path="..\\data\\train.csv"
Test_Path="..\\data\\test.csv"

class DataSet:
    # Declaring global variables
    _train_x=[]
    _train_y=[]
    _test_x=[]
    _test_y=[]
    _index_in_epochs=0
    _epochs_completed = 0
    _num_examples =0

    # defined the __init__ Constructor
    def __init__(self, data_path, class_list, Test=False):

        class_num= len(class_list)

        if Test:
           all_data= pd.read_csv(Test_Path, index_col=False)
           for i in range(0, len(all_data.label)):
               label= all_data.iloc[i,0]

               if label not in class_list: # eg [0,1,2,3,4,5,6,7,9]
                   continue

               self._test_x.append(all_data.iloc[i, 1:1201])

               # manually One hot Encoding labels
               loc= class_list.index(label)
               y=[0 for _ in range(class_num) ]
               y[loc] = 1
               self._test_y.append(y)

           else:
               fs=os.listdir(data_path)
               all_data= pd.DataFrame()

               for f in fs:
                   file_path = os.path.join(data_path, f)
                   # Append if its a csv file
                   if 'csv' in f:
                       data = pd.read_csv(file_path, index_col=False)
                       all_data = all_data.append(data)
                       np.random.shuffle(all_data.values)

               train_data = all_data[0:1000 * class_num]
               train_data.to_csv(Train_Path, index=False)

               test_data = all_data[1000 * class_num:]
               test_data.to_csv(Test_Path, index=False)

               self._num_examples = class_num * 1000

               for i in range(0,self._num_examples):
                   label= all_data.iloc[i,0]
                   if label not in class_list:
                       continue
# Store Data
                   self._train_x.append(all_data.iloc[i, 1:1201])

                    # One Hot Encode the labels
                   loc = class_list.index(label)
                   y = [0 for _ in range(class_num)]
                   y[loc] = 1
                   self._train_y.append(y)

               for i in range(self._num_examples, len(all_data.label)):
                   label = all_data.iloc[i, 0]
                   if label not in class_list:
                       continue

                   # Creating test Array
                   self._test_x.append(all_data.iloc[i, 1:1201])

                    # one hot encode the labels
                   loc = class_list.index(label)
                   y = [0 for i in range(class_num)]
                   y[loc] = 1
                   self._test_y.append(y)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_y(self):
        return self._train_y

    @property
    def test_x(self):
        return self._test_x

    @property
    def test_y(self):
        return self._test_y

    @property
    def index_in_epoch(self):
        return self._index_in_epochs

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        # Data sampling in batches of 50 and prevent Resampling Strategy
        start = self._index_in_epochs
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            if rest_num_examples != 0:
                x_rest_part = self.train_x[start:self._num_examples]
                y_rest_part = self.train_y[start:self._num_examples]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epochs
                x_new_part = self.train_x[start:end]
                y_new_part = self.train_y[start:end]
                batch_x, batch_y = np.concatenate(
                    (x_rest_part, x_new_part), axis=0), np.concatenate(
                    (x_rest_part, y_new_part), axis=0)
            else:
                # Start next epoch
                start = 0
                self._index_in_epochs = batch_size - rest_num_examples
                end = self._index_in_epochs
                batch_x = self.train_x[start:end]
                batch_y = self.train_y[start:end]
        else:
            self._index_in_epochs += batch_size
            end = self._index_in_epochs
            batch_x = self.train_x[start:end]
            batch_y = self.train_y[start:end]

        return np.array(batch_x), np.array(batch_y)

    def get_test_data(self):
        x= self.test_x
        y=self.test_y
        return np.array(x), np.array(y)

    def get_train_data(self):
        x = self.train_x
        y = self.train_y
        return np.array(x), np.array(y)

    def _normalization(self, data):

        data = data / (256 / 2.0) - 1
        return data

def main():
    dataset= DataSet(Data_Path, [0,3,6])
    x, y=  dataset.get_train_data()
    print(np.max(x))

if __name__ == '__main__':
    main()










