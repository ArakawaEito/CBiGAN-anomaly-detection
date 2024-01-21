import numpy as np
import pandas as pd
import math
import io
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# 画像保存用のヘルパー
def plot_to_image(figure):
    """matplotlibのプロットをPNGに変換する。
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    
    # buf内の「ファイルポインタ」（現在の読み書き位置）を指定した位置に移動しておく(0=先頭)
    buf.seek(0) 
    # バフに保存したPNG画像をtensor imageに変換
    image = tf.io.decode_png(buf.getvalue(), channels=3) # channels: カラーチャネルの数。4の場合、RGBA画像を出力
    image = tf.expand_dims(image, axis=0)# バッチの次元を追加
    
    return image

def plot_anomaly_score(start, end, freq, anomaly_score, label, lLmit, uLimit):
    """
    start:スタート時刻, 次の形式の文字列->'2022-09-21 06:00:00',
    end:終了時刻, 次の形式の文字列->'2022-09-21 22:00:00',
    freq:軸の頻度(時間間隔sec), int
    anomaly_score:freq単位の異常度の配列, 1D ndarray
    label:freq単位のラベルの配列, 1D ndarray
    lLmit, uLimit:y軸の下限、上限
    """
    # 音データとラベルデータの長さは異なるため揃える
    min_length = min(len(anomaly_score), len(label))
    anomaly_score = anomaly_score[:min_length]
    label = label[:min_length]

    freq_S = f"{freq}S"
    time_index = pd.date_range(start=start, end=end, freq=freq_S)
    time_index = time_index[:len(anomaly_score)]

    # # 最大値のインデックスを取得
    # max_index = np.argmax(anomaly_score)
    # print(f"異常度の最大時刻：{time_index[max_index]}")

    index_num_per_houre = 3600//freq
    graphnum = -(-len(time_index)//index_num_per_houre) # 切り上げ

    fig, axes = plt.subplots(graphnum, 1, figsize=(35, 300))
    for i in range(graphnum):

        axes[i].plot(date2num(time_index), (anomaly_score), label="異常度")
        axes[i].fill_between(date2num(time_index), label, color='red', alpha=0.2, label="異常ラベル")
        
        axes[i].set_xlabel('時刻', fontname="MS Mincho", fontsize=40)
        axes[i].set_ylabel('異常度', fontname="MS Mincho", fontsize=40)
        axes[i].xaxis.set_tick_params(labelsize = 35)
        axes[i].yaxis.set_tick_params(labelsize = 35)

        axes[i].legend(loc='lower center', bbox_to_anchor=(.5, 1.), ncol=2, prop={'family':"MS Mincho", 'size':35})

        # axes[i].set_xlabel('Time')
        # axes[i].set_ylabel('Abnormality')

        axes[i].set_ylim(lLmit, uLimit)
        if i<(graphnum-1):
            axes[i].set_xlim(
                date2num(time_index[index_num_per_houre*i]), date2num(time_index[index_num_per_houre*(i+1)]))
        else:
            axes[i].set_xlim(date2num(time_index[index_num_per_houre*i]), date2num(time_index[-1]))
            
        axes[i].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        #5分おきにラベル
        Minute1=mdates.MinuteLocator(range(60),5)   
        axes[i].xaxis.set_major_locator(Minute1)
    
    return fig