from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse # 控制超参数
import numpy as np
import os
import pandas as pd

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, daily_trend,add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param daily_trend:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    # y_trend: (epoch_size, output_length, num_nodes, 1)
    """

    num_samples, num_nodes = df.shape  #（num_samples, num_nodes）
    print('daily_trend shape',daily_trend.shape)
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        #time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(5, "m")
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1) #(num_samples, num_nodes, features)

    time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(5, "m")
    time_ind = time_ind.astype(int)

    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    y_trend = []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        y_trend_t = daily_trend[time_ind[t + y_offsets],...]
        x.append(x_t)
        y.append(y_t)
        y_trend.append(y_trend_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    y_trend = np.stack(y_trend, axis=0)
    return x, y, y_trend

def calculate_daily_trend(df):
    df = df.iloc[0:round(len(df)*0.7),:]  #只使用训练集的数据计算trend
    time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(5, "m")
    time_ind = time_ind.astype(int)
    df.index = time_ind

    trend_list = []
    for ind in range(max(time_ind)+1):
        mean_timeind = np.array(df[df.index == ind].mean())
        trend_list.append(mean_timeind)

    daily_trend = np.stack(trend_list,axis=0)

    return daily_trend



def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    print('num_samples:',len(df))
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),)) # 聚合
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))

    #calculate the daily_trends
    daily_trend = calculate_daily_trend(df)


    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y, y_trend = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        daily_trend = daily_trend,
        add_time_in_day=True,
        add_day_in_week=False, #在graph wavenet的实现代码中，all_day_in_week为True时维度为3
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape, ", y_trend shape", y_trend.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train, ytrend_train = x[:num_train], y[:num_train], y_trend[:num_train]
    # val
    x_val, y_val, ytrend_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
        y_trend[num_train: num_train + num_val],
    )
    # test
    x_test, y_test, ytrend_test = x[-num_test:], y[-num_test:], y_trend[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y, _ytrend = locals()["x_" + cat], locals()["y_" + cat], locals()["ytrend_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape, "ytrend:", _ytrend.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            ytrend=_ytrend,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    return df,daily_trend


def main(args):
    print("Generating training data")
    df,daily_trend = generate_train_val_test(args)# 生成训练集、验证集、测试集

    return df,daily_trend


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--output_dir", type=str, default="data/pems-bay/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="../data/pems-bay.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    df,daily_trend = main(args)
