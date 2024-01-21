import tensorflow as tf
import numpy as np

# 学習時のMetricとして使用する異常度。各ミニバッチの異常度の合計を返す。
@tf.function
def Anomaly_score(x, E_x, G_E_x, discriminator, Lambda):
    criterion_L1 = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE) # 異常スコア計測用(データごとに算出)->array
    
    _, x_feature = discriminator(x, E_x, training=False)
    _, G_E_x_feature = discriminator(G_E_x, E_x, training=False)
    
    residual_loss = tf.reduce_mean(tf.abs(x - G_E_x), axis=[1, 2, 3])
    discrimination_loss = criterion_L1(x_feature, G_E_x_feature) # feature-matching loss
    
    total_anomaly_score = (1-Lambda)*residual_loss + Lambda*discrimination_loss

    return total_anomaly_score


def test(encoder, generator, discriminator, test_loader, Lambda=0.1):
    """
    encoder         :エンコーダ
    generator       :ジェネレータ
    discriminator   :ディスクリミネータ
    test_loader     :データセット->tf.data
    Lambda          :異常度におけるdiscrimination_lossの比率
    =========================================================
    return:
    各データごとの異常度を要素とする一次元配列(ndarray)
    """
    all_anomaly_score  =[]
    for image_batch in test_loader:
        
        ## 異常度の計算 ##
        training=False
        E_x = encoder(image_batch, training=training)
        G_E_x = generator(E_x, training=training)

        anomaly_score = Anomaly_score(image_batch, E_x, G_E_x, discriminator, Lambda) # -> array
        all_anomaly_score += anomaly_score.numpy().tolist()


    return np.array(all_anomaly_score)

