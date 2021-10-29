import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from itertools import chain

acce_column_names = ['timestamp', 'x', 'y', 'z']
emg_column_names = ['timestamp', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']
gyro_column_names = ['timestamp', 'x', 'y', 'z']
ori_column_names = ['timestamp', 'x', 'y', 'z', 'w']
ori_euler_column_names = ['timestamp', 'roll', 'pitch', 'yaw']

# the 10 sample time each key
# we found whose timing from the local maximum of the sum emg figure
back_sample = [1.98, 3.36, 4.64, 5.905, 7.1, 8.205, 9.275, 10.315, 11.285, 12.33]
front_sample = [3.44, 4.91, 6.175, 7.56, 8.79, 10.225, 11.385, 12.51, 13.855, 15.03]
left_sample = [1.27, 2.565, 3.845, 5.045, 6.21, 7.38, 8.775, 10.145, 11.31, 12.45]
right_sample = [0.88, 2.005, 3.22, 4.385, 5.63, 6.825, 8.06, 9.18, 10.5, 11.86]
enter_sample = [1.605, 2.824, 3.8, 4.785, 5.91, 6.955, 7.945, 8.915, 9.855, 11.025]

keys = [('Backward', 1456704054, '-r', back_sample), ('Enter', 1456704184, '-b', enter_sample),
        ('Forward', 1456703940, '-k', front_sample),
        ('Left', 1456704106, '-g', left_sample), ('Right', 1456704146, '-m', right_sample)]

#             ( name,       col,        frequency , filter low pass,)
dataNames = [('accelerometer', acce_column_names, 50.0, 40),
             ('emg', emg_column_names, 200.0, 10),
             ('gyro', gyro_column_names, 50.0, 40),
             ('orientation', ori_column_names, 50.0, 40),
             ('orientationEuler', ori_euler_column_names, 50.0, 40)
             ]


# extract a sample
def extract_click(x, y, position, width):
    temp_x = x[int(position - width / 2):int(position + width / 2)]
    temp_y = y[int(position - width / 2):int(position + width / 2)]
    return temp_x, temp_y

#using band pass filter to reduce the noise
def filterSignal(time, emg, low_pass=10, sfreq=200, high_band=5, low_band=90):
    # normalise cut-off frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)

    # create bandpass filter for EMG
    b1, a1 = signal.butter(4, [high_band, low_band], btype='bandpass')

    # process EMG signal: filter EMG
    emg_filtered = signal.filtfilt(b1, a1, emg)

    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)

    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass / sfreq
    b2, a2 = signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = signal.filtfilt(b2, a2, emg_rectified)
    return emg_envelope


# sum all emg and find the 10 local max. The local max determine the 10 dataset.
def makeSumDataFigure():
    print('make sum Data figure...')
    for dataName, col_list, frequency, low_pass in dataNames:

        fig = plt.figure(num=None, figsize=(11, 6), dpi=80, facecolor='w', edgecolor='k')
        for key_name, id, color_s, click_pos in keys:
            fileName = "../Data/output/filtered/{0}_{1}.csv".format(key_name, dataName)
            data = pd.read_csv(fileName)
            df = pd.DataFrame(data, columns=col_list)
            temp_y = np.zeros(len(df.pop(col_list[0]).tolist()))

            x = np.array([i / frequency for i in range(0, len(temp_y), 1)])  # sampling rate 200 Hz

            for col_name in col_list[1:]:
                y = df.pop(col_name).tolist()
                for i in range(len(y)):
                    temp_y[i] = temp_y[i] + y[i]

            peaks, _ = signal.find_peaks(temp_y, distance=int(frequency*0.9))
            plt.plot(x, temp_y, color_s, linewidth=0.5, label=key_name)

            plt.plot(peaks / frequency, temp_y[peaks], "x")
            plt.text(0, 0, 'x:{},\n y:{}'.format(peaks / frequency, temp_y[peaks]), fontsize=9)
            plt.title(key_name + '_' + dataName )
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.savefig('../Figure/SumData/all{0}_{1}.png'.format(key_name, dataName), dpi=100)
            plt.cla()



# print and download figure
def makeFilteredFigure():
    print('make Filtered Figure...')
    fig = plt.figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.xlabel('x')
    plt.ylabel('y')

    for dataName, col_list, frequency, low_pass in dataNames:

        for key_name, id, color_s, click_pos in keys:
            fileName = "../Data/output/filtered/{0}_{1}.csv".format(key_name, dataName)
            data = pd.read_csv(fileName)
            df = pd.DataFrame(data, columns=col_list)

            for col_name in col_list[1:]:
                plt.title(key_name + '_' + dataName + '_' + col_name)
                y = df.pop(col_name).tolist()


                x = np.array([i / frequency for i in range(0, len(y), 1)])

                plt.plot(x, y, color_s, linewidth=0.5, label=key_name)


                plt.savefig('../Figure/Filtered/{0}_{1}_{2}.png'.format(key_name, dataName, col_name), dpi=100)
                plt.cla()


def makeSplittedFigure():
    print('make Splitted Figure...')
    fig = plt.figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.xlabel('x')
    plt.ylabel('y')

    for dataName, col_list, frequency, low_pass in dataNames:
        for key_name, id, color_s, click_pos in keys:
            for colName in col_list[1:]:
                plt.title(key_name + '_' + dataName + '_' + colName)
                for i in range(10):
                    fileName = "../Data/output/splitted/{0}_{1}_{2}.csv".format(key_name, dataName,i)
                    data = pd.read_csv(fileName)
                    df = pd.DataFrame(data, columns=col_list)

                    y = df[colName]
                    x = np.array([i / frequency for i in range(0, len(y), 1)])

                    plt.plot(x, y, color_s, c=np.random.rand(3, ), linewidth=0.5, label=key_name)


                plt.savefig('../Figure/splitted/{0}_{1}_{2}.png'.format(key_name, dataName, colName), dpi=100)
                plt.cla()


# normalize all data
def normalize():
    print('normalize data...')
    for dataName, col_list, frequence, low_pass in dataNames:
        dataSet = {}
        for key_name, id, color_s, click_pos in keys:
            fileName = '..\\Data\\output\\filtered\\{0}_{1}.csv'.format(key_name, dataName)
            data = pd.read_csv(fileName)
            dataSet[key_name] = pd.DataFrame(data, columns=col_list)

        for col in col_list[1:]:
            temp_data = []
            for dataFrame in dataSet:
                temp_data.append(list(dataSet[dataFrame][col]))
            max_value = np.max(list(chain.from_iterable(temp_data)))
            min_value = np.min(list(chain.from_iterable(temp_data)))
            for dataFrame in dataSet:
                norm_data = list((dataSet[dataFrame][col] - min_value) / (max_value - min_value))
                dataSet[dataFrame][col] = norm_data

        for dataFrame in dataSet:
            dataSet[dataFrame].to_csv('..\\Data\\output\\normalized\\{0}_{1}.csv'.format(dataFrame, dataName))


# spliting 10 data with the 10 local max
def splitData():
    print('spliting data...')
    for dataName, col_list, frequence, low_pass in dataNames:
        for key_name, id, color_s, click_pos in keys:
            fileName = "../Data/output/normalized/{0}_{1}.csv".format(key_name, dataName)
            data = pd.read_csv(fileName)
            df = pd.DataFrame(data, columns=col_list)

            for count, time in enumerate(click_pos):
                df2 = pd.DataFrame()
                for i, col_name in enumerate(col_list[1:]):

                    y = df[col_name].tolist()
                    x = np.array([i / frequence for i in range(0, len(y), 1)])  # sampling rate 200 Hz
                    index = round(time * frequence)
                    if key_name != 'emg':
                        index += 20
                    click_x, click_y = extract_click(x, y, index, frequence)
                    click_x = [temp_x - click_x[0] for temp_x in click_x]
                    df2.insert(i, column=col_name, value=click_y)
                df2.to_csv('../Data/output/splitted/{0}_{1}_{2}.csv'.format(key_name, dataName, count))



# use Bandpass filters to filter the noices
def filterData():
    print('start filter data...')
    for dataName, col_list, frequence, low_pass in dataNames:
        for key_name, id, color_s, click_pos in keys:
            fileName = "../Data/Myo Keyboard Data/{0}/{1}-{2}.csv".format(key_name, dataName, id)
            data = pd.read_csv(fileName)
            df = pd.DataFrame(data, columns=col_list)

            for i, col_name in enumerate(col_list[1:]):
                y = df[col_name].tolist()

                x = np.array([i / frequence for i in range(0, len(y), 1)])  # sampling rate 200 Hz
                y = filterSignal(x, y, low_pass=low_pass)

                df[col_name] = y
                df.to_csv('..\\Data\\output\\filtered\\{0}_{1}.csv'.format(key_name, dataName))



if __name__ == '__main__':
    filterData()
    makeSumDataFigure()
    makeFilteredFigure()
    normalize()
    splitData()
    makeSplittedFigure()
