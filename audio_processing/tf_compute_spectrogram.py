import numpy as np
import pandas as pd
from tqdm import tqdm 
import os
import pathlib
import shutil
import librosa as lb
import wave
import tensorflow as tf

# データ拡張のためのクラスをimport 
from .audio_DA import Multiple, ResampleWaveform, GaussianNoiseSNR, PinkNoiseSNR, PitchShift, TimeShift, VolumeShift


@tf.function
def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """
    パワースペクトログラム（振幅の二乗）をデシベル（dB）単位に変換する
    「10 * log10(S / ref)」のスケーリングを数値的に計算
    Based on:
    https://librosa.org/doc/0.10.1/_modules/librosa/core/spectrum.html#power_to_db
    ===============================================================================
    S:パワースペクトル(振幅の二乗)
    ref:Sをrefを基準にしてスケーリング
    amin:Sとrefの最小閾値
    top_db: スペクトログラムのダイナミックレンジ（最大値と最小値の差）を制御し、特に低い値（背景ノイズなど）を抑える。
            スペクトログラムにおける重要な信号の特徴を際立たせる。
            スペクトログラムの最大値からtop_db離れた値が下限になる
    """
    def _tf_log10(x):
        # 底の変換
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    
    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    # スペクトログラムの値を特定のデシベル範囲内に収める
    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
    # tf.print("powet to db")

    return log_spec

# @tf.function
# def compute_spectrogram(y, spec_params, duration, sr):
#     """
#     スペクトログラム変換の関数
#     """
#     # スペクトログラムの画像のサイズを統一するためにゼロパティングを実施
#     input_len = duration*sr
#     y = y[:input_len]
#     y = tf.cast(y, dtype=tf.float32)
#     zero_padding = tf.zeros([input_len] - tf.shape(y), dtype=tf.float32)
#     y = tf.concat([y, zero_padding], 0)

#     # 短時間フーリエ変換によりスペクトログラム算出
#     # spec_params = {'frame_length':256, 'frame_step':123}のときshapeが(129, 129)になる
#     spectrogram = tf.signal.stft(y, **spec_params)
#     # 振幅スペクトル
#     spectrogram = tf.abs(spectrogram)
#     # 対数振幅スペクトル
#     log_spectrogram = tf.math.log(spectrogram + 1e-7)

#     # 一般的なスペクトログラムの形式になるようチャンネル次元を追加し、次元を交換
#     log_spectrogram = log_spectrogram[..., tf.newaxis] # -> shape (`時間`, `周波数`, `channels`).
#     log_spectrogram = tf.transpose(log_spectrogram, perm=[1, 0, 2])# -> shape (`周波数`, `時間`, `channels`).

#     return log_spectrogram 

@tf.function
def compute_mel_spectrogram(y, spec_params, mel_params, duration, sr):
    """
    メルスペクトログラム変換の関数
    """
    # スペクトログラムの画像のサイズを統一するためにゼロパティングを実施
    input_len = duration*sr
    y = y[:input_len]
    y = tf.cast(y, dtype=tf.float32)
    zero_padding = tf.zeros([input_len] - tf.shape(y), dtype=tf.float32)
    y = tf.concat([y, zero_padding], 0)

    # 短時間フーリエ変換によりスペクトログラム算出
    # spec_params = {'frame_length':256, 'frame_step':123}のときshapeが(129, 129)になる
    spectrogram = tf.signal.stft(y, **spec_params)
    # パワースペクトル
    spectrogram = tf.abs(spectrogram)**2 # -> shape (`時間`, `周波数`).

    # メルスペクトログラム
    num_spectrogram_bins = spectrogram.shape[-1]  
    mel_basis = tf.signal.linear_to_mel_weight_matrix(num_spectrogram_bins=num_spectrogram_bins, sample_rate=sr, **mel_params)
    mel_spectrograms = tf.tensordot(spectrogram, mel_basis, 1)   

    # dbに変換
    db_mel_spectrograms = power_to_db(mel_spectrograms)

    # # 対数振幅スペクトル
    # log_mel_spectrogram = tf.math.log(mel_spectrograms + 1e-7)

    # 一般的なスペクトログラムの形式になるようチャンネル次元を追加し、次元を交換
    db_mel_spectrograms = db_mel_spectrograms[..., tf.newaxis] # -> shape (`時間`, `周波数`, `channels`).
    db_mel_spectrograms = tf.transpose(db_mel_spectrograms, perm=[1, 0, 2])# -> shape (`周波数`, `時間`, `channels`).

    return db_mel_spectrograms 



#正常音(または異常音)だけを切り抜いてメルスペクトログラムを作成しnpy形式に保存するためのクラス
class Sound_load_and_save_by_label():
    def __init__(self, sr, duration, spec_params, mel_params, way):
        '''
        sr:スペクトログラムに変換する音データのサンプリング周波数，すべてのデータのsrが揃っていない場合は揃える
        duration:メルスペクトログラムの時間幅
        spec_params:メルスペクトログラムの各種パラメータの辞書
        way:切り抜きの仕方．
            True:正常音の部分を切り抜く
            False: 異常音の部分を切り抜く
        '''
        self.sr = sr
        self.duration = duration
        self.spec_params = spec_params   
        self.mel_params = mel_params
        self.way = way
        
    # 正常尾を抜き出す
    def clipping_normal(self, label):
        '''
        label:1列の正解ラベルのデータフレーム(ラベルの列だけ)，0:正常, 1:異常
        戻り値:
            list_start:正常音の開始時間を格納したリスト
            list_duration:正常音の継続時間を格納したリスト

        '''
        list_start = [] # 正常音の開始時間(s)
        list_duration = []# 正常音の時間長(s)
        search= 1 if label.iloc[0, 0]==0 else 0 #なんのラベルを探すか（searchが1のときはラベルが1の行を探す）
        start = 0
        for num in range(len(label)):
            if search==0 and (label.iloc[num, 0]==search): 
                start=num
                search=1
            elif search==1 and label.iloc[num, 0]==search:
                stop=num-1
                duration = (stop-start+1)*0.2
                list_start.append(start*0.2)
                list_duration.append(duration)
                search=0
        
        if start>stop:
            stop=len(label)-1
            duration = (stop-start+1)*0.2
            list_start.append(start*0.2)
            list_duration.append(duration)

        return list_start, list_duration


    # 異常音を抜き出す
    def clipping_abnorm(self, label):
        '''
        label:1列の正解ラベルのデータフレーム(ラベルの列だけ)，0:正常, 1:異常
        戻り値:
            list_start:異常音の開始時間を格納したリスト
            list_duration:異常音の継続時間を格納したリスト

        '''
        list_start = [] # 異常音の開始時間(s)
        list_duration = []# 異常音の時間長(s)
        search= 1 if label.iloc[0, 0]==0 else 0 #なんのラベルを探すか（searchが1のときはラベルが1の行を探す）
        start = 0
        for num in range(len(label)):
            if search==1 and (label.iloc[num, 0]==search): 
                start=num
                search=0
            elif search==0 and label.iloc[num, 0]==search:
                stop=num-1
                duration = (stop-start+1)*0.2
                list_start.append(start*0.2)
                list_duration.append(duration)
                search=1
 
        if start>stop:
            stop=len(label)-1
            duration = (stop-start+1)*0.2
            list_start.append(start*0.2)
            list_duration.append(duration)

        return list_start, list_duration



    # npy形式でメルスペクトログラムを保存
    def load_and_save_by_label(self, record, out_dir, transform, label_data_df):
        '''
        recordからlabel_data_dfに基づき正常音(または異常音)を取り出しスペクトログラムに変換する
        ----------------------------------------------------------
        record:音声ファイルのパス
        out_dir:npyファイル出力するディレクトリのパス
        transform: audio_DA.pyにあるデータ拡張クラスをまとめたオブジェクト
        label_data_df:正解ラベルのデータフレーム
        '''
        duration = self.duration
        spec_params = self.spec_params
        mel_params = self.mel_params
        way = self.way

        mean = []
        std = []

        # ラベルデータをもとに正常音の部分だけを取り出す
        if way:
            list_start, list_duration = self.clipping_normal(label_data_df)
        else:
            list_start, list_duration = self.clipping_abnorm(label_data_df)

        count = 0 # ファイル名を付けるときに使用
        for offset, length in tqdm(zip(list_start, list_duration), total=len(list_start)):
            # 分割後，何個のフレームができるか
            num_cut = int(length//duration)
            for i in range(num_cut):       
                y, sr = lb.load(record, sr = None, offset=offset+(i*duration), duration=duration)
                # print(len(y))
                y_augmented = transform(y) # データ拡張        
                melspec = compute_mel_spectrogram(y, spec_params, mel_params, duration, sr).numpy()
                melspec_augmented = compute_mel_spectrogram(y_augmented, spec_params, mel_params, duration, sr).numpy()       
                mean.append(melspec.mean())
                std.append(melspec.std())

                #ファイル名の最初の文字列に、処理前のデータに”0”をつけ処理後は”1”をつけて区別する
                # record_name = '0'+'_'+str(count)+ '_' +record.split('/')[-1].replace('.wav', '.npy') 
                record_name = '0'+'_'+str(count)+ '_' +record.with_suffix('.npy').name
                augmented_record_name = '1'+record_name.replace('.wav', '.npy')

                np.save(f'{out_dir}/{record_name}', melspec)
                np.save(f'{out_dir}/{augmented_record_name}', melspec_augmented)

                count+=1

        return np.array(mean).mean(), np.array(std).mean() 

    def load_and_save(self, record, out_dir):
        '''
        recordをdurationごとに分割してすべてスペクトログラムに変換する
        ------------------------------------------------------------
        record:音声ファイルのパス
        out_dir:npyファイル出力するディレクトリのパス
        '''
        file_basename = os.path.splitext(os.path.basename(record))[0]
        record = str(record)
        
        with wave.open(record) as wav:
            # wavデータのサンプル数を取得
            num_samples = wav.getnframes()
            # サンプリング周波数 [Hz] を取得
            sampling_frequency = wav.getframerate()      
            # 長さ
            total_time = num_samples // sampling_frequency     
            # 分割後，何個のフレームができるか
            num_cut = int(total_time//self.duration)
            print(f'{total_time}[sec]')

        mean = []
        std = []
        for i in tqdm(range(num_cut)):
            record_name = str(i)+ '_' + file_basename+'.npy'
            y, sr = lb.load(record, sr = None, offset=i*self.duration, duration=self.duration)
            melspec = compute_mel_spectrogram(y, self.spec_params, self.mel_params, self.duration, sr).numpy()
            mean.append(melspec.mean())
            std.append(melspec.std())
            np.save(f'{out_dir}/{record_name}', melspec)

        return np.array(mean).mean(), np.array(std).mean() 

def wav_to_spectrogram(parent_npy_output, soundData, sr, duration, spec_params, mel_params, transformName=[], list_df_labelData=None, overwrite=False, way=True):
    """
    parent_npy_output:npyファイル出力するディレクトリの親となるディレクトリ, 以下のような階層
        -parent_npy_output
            -npy_files/<npyファイル>
            -mean.npy
            -std.npy
            -train_mean_std.txt
    soudData:音声ファイルのパスのリスト
    list_df_labelData:ラベルデータのDFのリスト
        -None: soundDataをdurationごとに分割してすべてスペクトログラムに変換する
    transformName: 実施するデータ拡張の名前リスト
        -'GaussianNoiseSNR' :ホワイトノイズ付与
        -'PinkNoiseSNR'     :ピンクノイズ付与
        -'PitchShift'       :ピッチシフト
        -'TimeShift'        :タイムシフト
        -'VolumeShift'      :音量シフト
    overwrite:npy_outputが既に存在する場合に上書きするかどうか
    way:切り抜きの仕方．
        -True:正常音の部分を切り抜く
        -False: 異常音の部分を切り抜く
    ==============================================
    return:
    すべてのメルスペクトログラムの平均値と標準偏差
    """
    # データ拡張用オブジェクトのリスト
    list_transform = []
    if 'GaussianNoiseSNR' in transformName:
        list_transform.append(GaussianNoiseSNR(min_snr=15, max_snr=30))
    if 'PinkNoiseSNR' in transformName:
        list_transform.append(PinkNoiseSNR(min_snr=8, max_snr=30))
    if 'PitchShift' in transformName:
        list_transform.append(PitchShift(max_steps=2, sr=sr))
    if 'TimeShift' in transformName:
        list_transform.append(TimeShift(sr=sr))
    if 'VolumeShift' in transformName:
        list_transform.append(VolumeShift(mode="cosine"))
    # print(list_transform)
    transform = Multiple(list_transform)
    save_npy = Sound_load_and_save_by_label(sr, duration, spec_params, mel_params, way)

    parent_npy_output = pathlib.Path(parent_npy_output)
    npy_output = parent_npy_output/'npy_files'
    #npyファイルに変換
    print('npy_output:', npy_output)
    if(os.path.isdir(parent_npy_output) == True):
        print(f'{parent_npy_output}が既に存在します')
        if overwrite == False:
            print('上書せずに終了')
            all_mean = np.load(parent_npy_output/'mean.npy')
            all_std = np.load(parent_npy_output/'std.npy')

            return all_mean, all_std 
        else:
            print(f'{parent_npy_output}を削除します')
            shutil.rmtree(parent_npy_output)
            print(f'{parent_npy_output}削除完了')
    os.makedirs(npy_output, exist_ok=False)

    all_data_mean_std =[]
    if list_df_labelData is not None:  
        for i,j,k,l in zip(soundData, [npy_output]*len(soundData), [transform]*len(soundData), list_df_labelData):
            ret = save_npy.load_and_save_by_label(i, j, k, l) # -> タプル(mean, std)
            all_data_mean_std.append(ret)
    else:
        for record, output in zip(soundData, [npy_output]*len(soundData)):
            ret = save_npy.load_and_save(record, output)
            all_data_mean_std.append(ret)

    # # retにはsave_npy.load_and_save_by_labelの戻り値を要素とするlistが入る
    # ret = joblib.Parallel(n_jobs=-1)(joblib.delayed(save_npy.load_and_save_by_label)(i,j,k,l) for i,j,k,l in tqdm(zip(soundData, [npy_output]*len(soundData), [transform]*len(soundData), list_df_labelData), total=len(soundData)))
    # print(f'{npy_output}にnpyファイルを保存しました。')

    mean = [data[0] for data in all_data_mean_std]
    std = [data[1] for data in all_data_mean_std]
    all_mean, all_std = np.array(mean).mean(), np.array(std).mean() 
    np.save(parent_npy_output/'mean.npy', all_mean)
    np.save(parent_npy_output/'std.npy', all_std)
    with open(parent_npy_output/"train_mean_std.txt", mode='w') as f:
        f.write(f"mean:{all_mean}")
        f.write(f'std:{all_std}')

    return all_mean, all_std 
