import tensorflow as tf
import numpy as np


@tf.function
def resize_and_rescale(image, image_size, mean, std):
    """
    画像のリサイズとスケールの変換
    ==================================
    image->Tensor. shape(batch, h, w, channel)
    """
    # tf.print("from resize_and_rescale", tf.shape(image))
    # tf.print("before resize max", tf.reduce_max(image))
    # tf.print("before resize min", tf.reduce_min(image))

    # shape (`batch, 周波数`, `時間`, `channels`)の各次元を[1, 1, 1, 3]ずつ複製する。
    image = tf.tile(image, [1, 1, 1, 3]) #(`batch, 周波数`, `時間`, 1) => (`batch, 周波数`, `時間`, 3)
    image = tf.image.resize(image, image_size)

    # 8ビットスケール(0-255)に変換
    image = specTo8bit(image, mean=mean, std=std) #->EagerTensor

    # 正規化
    image = tf.cast(image,tf.float32)
    image =  (image - 127.5) / 127.5  # Normalize the images to [-1, 1]
    # image = (image / 255.0)           # Normalize the images to [0, 1]

    # tf.print("after resize max", tf.reduce_max(image))
    # tf.print("after resize min", tf.reduce_min(image))

    return image

@tf.function
def specTo8bit(X,  mean, std, eps=tf.constant(1e-6, dtype=tf.float32)):
    '''
    スペクトルのスケールを0~255に変換
    ---------------------------------
    mean->Tensor
    std->Tensor
    X->Tensor. shape(batch, h, w, channel)
    eps->Tensor
    '''
    # tf.print("mean", type(mean))
    # tf.print("from monotocolor", tf.shape(X))

    Xstd = (X - mean) / (std + eps) # epsでゼロ除算を防ぐ
    
    # _min, _max = tf.reduce_min(Xstd), tf.reduce_max(Xstd)

    # バッチ内の各データごとに最小値と最大値を計算
    _min = tf.reduce_min(Xstd, axis=[1, 2, 3], keepdims=True) # ->shape(batch, 1, 1, 1)
    _max = tf.reduce_max(Xstd, axis=[1, 2, 3], keepdims=True) # ->shape(batch, 1, 1, 1)

    # if (_max - _min) > eps:
    #     V = tf.clip_by_value(Xstd, _min, _max)
    #     V = 255 * (V - _min) / (_max - _min)
    #     V = tf.cast(V, tf.uint8)
    # else: 
    #     V = tf.zeros_like(Xstd, dtype=tf.uint8)

    # バッチ内の各データごとに正規化
    V = tf.where((_max - _min) > eps, 
                 255 * (tf.clip_by_value(Xstd, _min, _max) - _min) / (_max - _min), 
                 tf.zeros_like(Xstd))

    # uint8にキャストしておくことで確実に0~255の範囲に収まるようにする
    V = tf.cast(V, tf.uint8)

    return V

# def process_data(data, mean, std, image_size):
#     # tf.print(tf.shape(data))
#     # data = MonoToColor(data, mean=mean, std=std) #->EagerTensor
#     data = resize_and_rescale(data, image_size, mean, std) #->EagerTensor

#     return data

def process_path(file_path):
    file_path = file_path.numpy().decode('utf-8')
    data = np.load(file_path) # -> ndarray, shape(h, w, channel)

    return data

def return_dataset_loader(npy_files_paths, image_size, training, mean, std, batch_size=256):
    """
    image_size:スペクトログラムのリサイズのサイズ
    npy_files_paths:全npyファイルのパスが格納されたリスト
    training: Falseにするとデータセットの順番を決定的にする。
    """
    AUTOTUNE = tf.data.AUTOTUNE
    npy_count = len(npy_files_paths) # データ数
    
    # ファイル名を生成するデータセット
    path_list_ds = tf.data.Dataset.from_tensor_slices(npy_files_paths)
    # for i in path_list_ds:
    #     # tf.print(type(i.numpy()))
    
    # テスト時はデータの順番が保証されている必要があるため条件分岐する
    if training==True:
        # ファイル名から画像データを生成するデータセット(cache->shuffleの順番でないとshuffleの意味がなくなる)
        ds = path_list_ds.map(lambda path: tf.py_function(func=process_path, inp=[path], Tout=tf.float32), num_parallel_calls=AUTOTUNE).cache().shuffle(npy_count).batch(batch_size).map(lambda x: resize_and_rescale(x, image_size, mean, std), num_parallel_calls=AUTOTUNE)

    elif training==False:
        # ファイル名から画像データを生成するデータセット
        # deterministic=True:出力の順番を決定的にする
        ds = path_list_ds.map(lambda path: tf.py_function(func=process_path, inp=[path], Tout=tf.float32), deterministic=True).cache().batch(batch_size).map(lambda x: resize_and_rescale(x, image_size, mean, std), deterministic=True)
        
    # バッチに分割する
    ds = ds.prefetch(buffer_size=3)
    
    return ds