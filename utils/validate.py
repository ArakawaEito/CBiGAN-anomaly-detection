import os
import math
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve, auc

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


def convert_label_unit(label, duration, list_anomalyLabel):
    '''
    0.2秒単位の配列をduration単位の配列に変換するともに、異常ラベルを1に統一する
    各durationごとにラベルを分割したときに、分割した要素の中に異常ラベルがあればその時間帯のラベルを異常とする
    ---------------------------------------------
    label: 1次元のラベルの配列, ndarray 
    duration: 変換後のラベルの単位
    list_anomalyLabel:異常に対応するラベルのリスト
    '''
    unit_sample = int(duration/0.2)
    num_unit = int(len(label)/unit_sample)
    
    # 各unitごとのラベル(ndarray)を要素とするリスト
    label_each_duration = [label[i*unit_sample:(i+1)*unit_sample] for i in range(num_unit)]

    # 各unitごとのラベル(ndarray)に一つでも異常ラベルがあればそのunitのラベルを1とする
    label_unit = np.array([1 if any(x in unit for x in list_anomalyLabel) else 0 for unit in label_each_duration])

    return label_unit

def validate(label, anomaly_score, thr, metrics):
    """
    異常ラベルが付与されている区間のうち異常と判断されたデータを含む区間の割合を再現率とする評価関数
    ==========================================================================================
    thr             :閾値
    anomaly_score   :異常度の配列, ndarray 
    label          :異常1, 正常0の1次元のラベルの配列, ndarray 
    metrics         :評価指標のリスト
        -'Precision':適合率
        -'Recall'   :再現率
        -'FPR'      :偽陽性率
    -------------------------------------------------------------------------------------------
    metricsのリストで指定した評価指標を要素とするリスト, 要素の順番はmetricsと同じ
    """
    test_v = pd.DataFrame(label, columns=['label'])
    test_v['z']=np.where(anomaly_score>=thr, 1, 0) #'z'カラムに予測値を格納する
    test_v.reset_index(inplace=True, drop=True)

    ret_metrics = []
    for metric in metrics:
        assert metric in ['Precision', 'Recall', 'FPR'] # 引数metircsの指定が正しいか確認
        if metric=='Precision':
            # 適合率
            tp = test_v[(test_v['label']==1)&(test_v['z']==1)] #tp: TPのインデックス
            z_p = test_v[test_v['z']==1] #予測値がPositiiveのインデックス
            pre_score=len(tp)/len(z_p)
            ret_metrics.append(pre_score)
        
        elif metric=='Recall':
            # 再現率
            df_anomaly_score=[] #異常音の帯（異常音の範囲）（データフレーム）を集めるリスト
            search= 1 if test_v.loc[0, 'label']==0 else 0 #なんのラベルを探すか（searchが1のときはラベルが1(または2)の行を探す）
            start = 0
            for num in range(len(test_v)):
                if search==1 and (test_v.loc[num, 'label']==1): #学習データのラベルが1のとき異常
                    start=num
                    search=0
                elif search==0 and test_v.loc[num, 'label']==search:
                    stop=num-1
                    anomaly_score_range=test_v.loc[start:stop].copy() #異常音の範囲のデータフレーム
                    df_anomaly_score.append(anomaly_score_range)
                    search=1

            if start>stop:
                anomaly_score_range=test_v.loc[start:].copy()
                df_anomaly_score.append(anomaly_score_range)    
            
            num_positive = 1 # 異常音の帯の中の何点以上を「異常音」と予測した時にその異常音の帯を異常音として判断したことにするか
            count=[] #異常音の帯の中に異常と判断した点がnum_positive以上ある異常音の帯を格納するリスト
            for i in range(len(df_anomaly_score)):
                if len(df_anomaly_score[i].loc[df_anomaly_score[i]['z']==1])>=num_positive: #異常音の帯の中に異常と判断した点がnum_positive以上の場合:
                    count.append(i)    
        #     print(len(df_anomaly_score))
            re_score=len(count)/len(df_anomaly_score)
            ret_metrics.append(re_score)

        elif metric=='FPR':
            # 偽陽性率
            fp = test_v[(test_v['label']==0)&(test_v['z']==1)] #fp: FPのインデックス
            label_n = test_v[test_v['label']==0] # 実測値がNegatiiveのインデックス
            fpr_score = len(fp)/len(label_n)
            ret_metrics.append(fpr_score)

    # print('適合率：%f'%(pre_score))    
    # print('再現率：%f'%(re_score))
    # print('偽陽性率：%f'%(fpr_score))
    
    return ret_metrics


def plot_PR(label, score, bins):
    """
    PR曲線をplotする
    ==================================================
    label   :異常1, 正常0の1次元のラベルの配列, ndarray 
    score   :異常度の配列, ndarray 
    bins    :PR曲線の閾値の数(計算時に切り捨てをしているので正確にこの数にはならない)
    """
    # print('label:', label)
    # print('score:', score)
    # print('len(label):', len(label))
    # print('len(score):', len(score))
    # print(np.bincount(label))  

    # 音データとラベルデータの長さは異なるため揃える
    min_length = min(len(score), len(label))
    score = score[:min_length]
    label = label[:min_length]

    assert len(score) == len(label)
    
    _, _, thresholds = precision_recall_curve(label, score)

    # 閾値を間引く
    interval = int(len(thresholds)/bins)
    thr = thresholds[:-1][::interval]
    # print(thresholds[-1])
    # thrにthresholdsの最後の要素を追加
    thr =  np.append(thr, thresholds[-1])
    # print(thr)
    precision = np.zeros(len(thr))
    recall = np.zeros(len(thr))
    f_score = np.zeros(len(thr))
    for i in range(len(thr)):
        precision[i], recall[i] = validate(label, score, thr[i], metrics=['Precision', 'Recall'])
        # F値
        if precision[i] + recall[i] == 0:
            f_score[i] = 0
        else:
            f = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            f_score[i] = f
        
    # f値が最大となるときの閾値
    idx_max_f_score = np.argmax(f_score)
    thr_max_f_score = thr[idx_max_f_score]

    auc_score = auc(recall, precision)
    # print(f'auc:{auc_score}')
        
    # PR曲線
    fig, ax = plt.subplots(facecolor="w", figsize=(5, 5))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f')) # 軸メモリの桁数
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f')) #軸メモリの桁数
    ax.grid()
    ax.plot(recall, precision)
    ax.set_xlabel('再現率', fontname="MS Mincho", fontsize=15)
    ax.set_ylabel('適合率', fontname="MS Mincho", fontsize=15)
    
    return auc_score, fig, thr_max_f_score, f_score


def plot_ROC(label, score, bins):
    """
    ROC曲線をplotする
    ==================================================
    label   :異常1, 正常0の1次元のラベルの配列, ndarray 
    score   :異常度の配列, ndarray 
    bins    :PR曲線の閾値の数(計算時に切り捨てをしているので正確にこの数にはならない)
    """
    # print('label:', label)
    # print('score:', score)
    # print('len(label):', len(label))
    # print('len(score):', len(score))
    # print(np.bincount(label))  

    # 音データとラベルデータの長さは異なるため揃える
    min_length = min(len(score), len(label))
    score = score[:min_length]
    label = label[:min_length]

    assert len(score) == len(label)
    
    _, _, thresholds = roc_curve(label, score)

    # 閾値を間引く
    interval = int(len(thresholds)/bins)
    thr = thresholds[:-1][::interval]
    # print(thresholds[-1])

    # thrにthresholdsの最後の要素を追加
    thr =  np.append(thr, thresholds[-1])
    # print(thresholds)    
    fpr = np.zeros(len(thr))
    recall = np.zeros(len(thr))
    for i in range(len(thr)):
        recall[i], fpr[i] = validate(label, score, thr[i], metrics=['Recall', 'FPR'])
    auc_score = auc(fpr, recall)
    # print(f'auc:{auc_score}')
        
    # ROC曲線
    fig, ax = plt.subplots(facecolor="w", figsize=(5, 5))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f')) # 軸メモリの桁数
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f')) #軸メモリの桁数
    ax.grid()
    ax.plot(fpr, recall)
    ax.set_xlabel('false positive rate', fontsize=15)
    ax.set_ylabel('true positive rate', fontsize=15)
    
    return auc_score, fig


# 等価騒音レベルを求める関数
def equivalentSoundLevel(x):
    '''
    x:騒音データ(ndarray)
    '''
    t = len(x)
    y = np.zeros(t)
    for i in range(t):
        y[i] = 10**(x[i]/10)
    mean_y = np.mean(y)
    Laeq =10*np.log10(mean_y)
    
    return Laeq

