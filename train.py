import matplotlib.pyplot as plt
import io
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers

class Train_Pipeline:
    def __init__(self, summary_writer, gp_weight=10.0, alpha=1e-4):
        """
        summary_writer:Tensor BoardのSummaryWriterオブジェクト
        gp_weight       :gradient_penaltyの重み
        alpha           :損失関数におけるL1 Lossの割合
        """
        self.summary_writer = summary_writer
        self.alpha = alpha
        self.gp_weight = gp_weight

        # optimizer
        self.G_E_optimizer = None
        self.D_optimizer = None

        # loss
        self.criterion= tf.keras.losses.BinaryCrossentropy()
        self.criterion_L1 = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)

        # metrics
        self.D_loss_metric = tf.keras.metrics.Mean()
        self.G_E_loss_metric = tf.keras.metrics.Mean()
        self.anomalyScore_metric = tf.keras.metrics.Mean()
        self.val_anomalyScore_metric = tf.keras.metrics.Mean() 

        # モデル 
        self.encoder = None
        self.generator = None
        self.discriminator = None
        
    def train(self, E, G, D, G_E_optimizer, D_optimizer, EPOCHS, BATCH_SIZE, EMBED_SIZE, train_loader, val_loader, ckpt, manager):
        """
        E           :エンコーダ
        G           :ジェネレータ
        D           :ディスクリミネータ
        G_E_optimizer: GとEのoptimizer
        D_optimizer: Dのoptimizer
        EMBED_SIZE  :潜在空間の次元数
        train_loader:データセット->tf.data
        val_loader: バリデーションデータセット->tf.data
        ckpt        :チェックポイントのオブジェクト, ckptはepoch回数を保持するstep属性を持つ必要がある ->tf.train.Checkpoint
        manager     :チェックポイントマネージャー ->tf.train.CheckpointManager 
        """
        # モデル
        self.encoder = E
        self.generator = G
        self.discriminator = D

        # optimizer
        self.G_E_optimizer = G_E_optimizer
        self.D_optimizer = D_optimizer

        best_val = float('inf')# チェックポイント更新時に使う最良値
        list_D_loss, list_G_E_loss, list_anomalyScore, list_val_anomalyScore = [], [], [], []
        tf.summary.trace_on(graph=True)
        for epoch in range(EPOCHS):
            print(f'\nstart of epoch {epoch}')
            # プログレスバー 
            num_iteration = int(train_loader.cardinality())
            with tqdm(total=num_iteration, unit="batch") as pbar:
                pbar.set_description(f"Epoch[{epoch}/{EPOCHS}]")
                       
                for step, image_batch in enumerate(train_loader):
                    
                    # generatorへの入力値 (batch_size, 1, 1, EMBED_SIZE)
                    z = tf.random.normal(shape=[tf.shape(image_batch)[0], EMBED_SIZE], mean=0.0, stddev=0.1)
                    # discriminatorに渡す画像データに加えるノイズ
                    stddev = 0.1 * (EPOCHS - epoch) / EPOCHS
                    noise1 = tf.random.normal(shape=tf.shape(image_batch), mean=0.0, stddev=stddev) # =>image_batchと同じ形状の乱数
                    noise2 = tf.random.normal(shape=tf.shape(image_batch), mean=0.0, stddev=stddev)        
                    
                    # 本物1,偽物0のラベルを作成
                    y_true = tf.ones([tf.shape(image_batch)[0]])
                    y_fake = tf.zeros([tf.shape(image_batch)[0]])
                    ## 訓練 ##
                    D_loss, G_E_loss = self.train_step(image_batch, z, noise1, noise2, y_true, y_fake)
                    if epoch==0 and step==0:
                        with self.summary_writer.as_default():
                            tf.summary.trace_export(
                            name="my_func_trace",
                            step=0)  
                            
                    self.D_loss_metric.update_state(D_loss)
                    self.G_E_loss_metric.update_state(G_E_loss)
                    
                    ## 異常度の計算 ##
                    E_x_val = self.encoder(image_batch, training=False)
                    G_E_x_val = self.generator(E_x_val, training=False)
                    self.anomalyScore_metric.update_state(self.Anomaly_score(image_batch, E_x_val, G_E_x_val))
                    
                    ## 生成画像の保存 ##
                    if step == 0:
                        ## 乱数からの生成画像 ##
                        save_image_size_for_z = min(BATCH_SIZE, 8)
                        save_z = z[:save_image_size_for_z]
                        G_z = self.generator(save_z, training=False)
                        figure_G_z = Train_Pipeline.image_grid(G_z, nrow=2, ncol=4)  
                        # figureを閉じないと次のimage_gridとくっついてしまって挙動がおかしくなるため、直後にplot_to_imageを呼び出して画像(tensor)に変換しておく
                        image_G_z = Train_Pipeline.plot_to_image(figure_G_z) 
                        
                        ## 再構成画像（G(E(x))） ##
                        save_image_size_for_recon = min(BATCH_SIZE, 4)
                        images = image_batch[:save_image_size_for_recon]
                        G_E_x = self.generator(self.encoder(images), training=False)
                        diff_images = tf.abs(images - G_E_x)
                        comparison = tf.concat([images , G_E_x, diff_images], axis=0) # バッチ方向に連結
                        # 各行がimages, G(E(x)), diff_imagesになるように描画・保存
                        figure_G_E_x = Train_Pipeline.image_grid(comparison, nrow=3, ncol=save_image_size_for_recon)   
                        image_G_E_x = Train_Pipeline.plot_to_image(figure_G_E_x)
                    
                    # プログレスバーの更新
                    pbar.set_postfix({"G_E_loss":self.G_E_loss_metric.result().numpy(), "D_loss":self.D_loss_metric.result().numpy(), "anomaly_score":self.anomalyScore_metric.result().numpy()})
                    pbar.update(1)
                    

            """ エポック終了時の処理 """
            # バリデーションデータの異常度算出および再構成画像の保存
            for val_step, val_image_batch in enumerate(val_loader):
                if val_step == 0:
                    # バリデーションデータに対する再構成画像（G(E(x))
                    save_image_size_for_recon = min(BATCH_SIZE, 4)
                    images = val_image_batch[:save_image_size_for_recon]
                    G_E_x = self.generator(self.encoder(images), training=False)
                    diff_images = tf.abs(images - G_E_x)
                    comparison = tf.concat([images , G_E_x, diff_images], axis=0) # バッチ方向に連結
                    # 各行がimages, G(E(x)), diff_imagesになるように描画・保存
                    val_figure_G_E_x = Train_Pipeline.image_grid(comparison, nrow=3, ncol=save_image_size_for_recon)   
                    val_image_G_E_x = Train_Pipeline.plot_to_image(val_figure_G_E_x)

                # バリデーションデータに対する異常度算出
                E_x_val = self.encoder(val_image_batch, training=False)
                G_E_x_val = self.generator(E_x_val, training=False)
                self.val_anomalyScore_metric.update_state(self.Anomaly_score(val_image_batch, E_x_val, G_E_x_val))

            # lossとmetricsの算出・表示
            loss_ge_mean = self.G_E_loss_metric.result()
            loss_d_mean = self.D_loss_metric.result()
            anomaly_score_mean = self.anomalyScore_metric.result()
            val_anomaly_score_mean = self.val_anomalyScore_metric.result()
            print(f"{epoch}/{EPOCHS} Epoch G_E_loss: {loss_ge_mean:.3f} D_loss: {loss_d_mean:.3f} anomaly_score: {anomaly_score_mean:.3f} val_anomaly_score: {val_anomaly_score_mean:.3f}")        

            # チェックポイントの更新
            ckpt.step.assign_add(1)
            if val_anomaly_score_mean < best_val:
                best_val = val_anomaly_score_mean
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))  

            # metricsのリセット
            self.G_E_loss_metric.reset_states()
            self.D_loss_metric.reset_states()
            self.anomalyScore_metric.reset_states()
            self.val_anomalyScore_metric.reset_states()
            
            list_G_E_loss.append(loss_ge_mean.numpy())
            list_D_loss.append(loss_d_mean.numpy())
            list_anomalyScore.append(anomaly_score_mean.numpy())
            list_val_anomalyScore.append(val_anomaly_score_mean.numpy())

            with self.summary_writer.as_default():
                tf.summary.scalar('G_E_loss', loss_ge_mean, step=epoch)
                tf.summary.scalar('D_loss', loss_d_mean, step=epoch)
                tf.summary.scalar('anomaly_score', anomaly_score_mean, step=epoch)
                tf.summary.scalar('val_anomaly_score', val_anomaly_score_mean, step=epoch)
                
                for weight in self.encoder.trainable_variables + self.generator.trainable_variables + self.discriminator.trainable_variables:
                    tf.summary.histogram(name=weight.name, data=weight, step=epoch)

                tf.summary.image("G(z)", image_G_z, step=epoch)
                tf.summary.image("G(E(x))", image_G_E_x, step=epoch)
                tf.summary.image("val_G(E(x))", val_image_G_E_x, step=epoch)

        history = {"G_E_loss":list_G_E_loss, "D_loss":list_D_loss, "anomalyScore": list_anomalyScore, "val_anomalyScore":list_val_anomalyScore}

        return history


    #Lipschitz制約を満たすための勾配ペナルティ
    @tf.function
    def gradient_penalty(self, x, x_gen, z, z_gen, training):
        batch_size = tf.shape(x)[0]
        z_epsilon = tf.random.uniform((batch_size, 1), 0.0, 1.0) # -> (batch_size, 1)
        x_epsilon = tf.reshape(z_epsilon, [batch_size, 1, 1, 1]) # -> (batch_size, 1, 1, 1)
        # 補間画像と補間変数の作成
        z_hat = z_epsilon * z + (1 - z_epsilon) * z_gen        
        x_hat = x_epsilon * x + (1 - x_epsilon) * x_gen


        with tf.GradientTape() as t:
            t.watch([x_hat, z_hat]) # tf.Tensorをtapeの監視対象に入れる（デフォルトだと入らない）
            score_hat = self.discriminator(x_hat, z_hat, training=training)
        dx, dz = t.gradient(score_hat, [x_hat, z_hat]) # -> type list length:2

        # batch次元を残して平坦化
        dx = tf.reshape(dx, (batch_size, -1)) # ->(batch_size, image_dim)
        dz = tf.reshape(dz, (batch_size, -1)) # -> (batch_size, latent_dim)

        grads = tf.concat((dx, dz), axis=1) # -> (batch_size, image_dim + latent_dim)

        # 勾配"ベクトル"のL2ノルムを計算
        grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1)) # -> (batch_size, )
        # L2ノルムと1.0の差の二乗をペナルティとする
        gradient_penalty = tf.reduce_mean(tf.square(grads_norm - 1.0))  # mean on batch

        return gradient_penalty 

    @tf.function
    def train_step(self, image_batch, z, noise1, noise2, y_true, y_fake):
        with tf.GradientTape() as G_E_tape, tf.GradientTape() as D_tape:
            training=True

            with tf.name_scope('FP'):
                E_x = self.encoder(image_batch, training=training) 
                G_z = self.generator(z, training=training)

                # 本物画像にノイズを加算した値とエンコーダの出力を渡す
                pred_true, _ = self.discriminator(image_batch + noise1, E_x, training=training)
                # 偽物画像にノイズを加算した値と乱数を渡す
                pred_fake, _ = self.discriminator(G_z + noise2, z, training=training)      

            with tf.name_scope('Loss'):
                """ discriminator wasserstein loss """ # => min
                D_loss = tf.reduce_mean(pred_fake - pred_true)
                # gradient penalty regularization
                gradient_penalty_loss = self.gradient_penalty(image_batch, G_z, z, E_x, training=training)
                # wasseserstein + GP
                D_total_loss = D_loss + self.gp_weight * gradient_penalty_loss


                """ generater and encoder loss """ #　=> min
                G_E_x = self.generator(E_x, training=training)
                E_G_z = self.encoder(G_z, training=training)

                G_E_loss = tf.reduce_mean(pred_true - pred_fake)

                # consistency losses
                consistency_loss = self.criterion_L1(image_batch, G_E_x) + self.criterion_L1(z, E_G_z) # L1 Loss
                # total loss
                G_E_total_loss = (1-self.alpha)*G_E_loss + self.alpha*consistency_loss


        with tf.name_scope('BP'):
            G_E_variables = self.generator.trainable_variables + self.encoder.trainable_variables # =>list型 = list型 + list型
            gradients_of_D = D_tape.gradient(D_total_loss, self.discriminator.trainable_variables)
            gradients_of_G_E = G_E_tape.gradient(G_E_total_loss, G_E_variables)

            self.D_optimizer.apply_gradients(zip(gradients_of_D, self.discriminator.trainable_variables)) 
            self.G_E_optimizer.apply_gradients(zip(gradients_of_G_E, G_E_variables))
        
        return D_total_loss, G_E_total_loss
    

    # 学習時のMetricとして使用する異常度。各ミニバッチの異常度の合計を返す。
    def Anomaly_score(self, x, E_x, G_E_x, Lambda=0.1):
        # criterion_L1 = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM) # 異常スコア計測用(バッチ方向に加算)
        
        _, x_feature = self.discriminator(x, E_x, training=False)
        _, G_E_x_feature = self.discriminator(G_E_x, E_x, training=False)
        
        residual_loss = self.criterion_L1(x, G_E_x)
        discrimination_loss = self.criterion_L1(x_feature, G_E_x_feature) # feature-matching loss
        
        total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss

        return total_loss
    
    # 画像保存用のヘルパー
    @staticmethod
    def plot_to_image(figure):
        """matplotlibのプロットをPNGに変換する。
        """
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # notebookに表示されているグラフを消すと同時にリソースを開放する.
        plt.close(figure)
        
        # buf内の「ファイルポインタ」（現在の読み書き位置）を指定した位置に移動しておく(0=先頭)
        buf.seek(0) 
        # バフに保存したPNG画像をtensor imageに変換
        image = tf.io.decode_png(buf.getvalue(), channels=3) # channels: カラーチャネルの数。4の場合、RGBA画像を出力
        image = tf.expand_dims(image, axis=0)# バッチの次元を追加
        
        return image
    
    @staticmethod
    def image_grid(images, nrow=4, ncol=4):
        """
        グリッド画像のfigを返す
        images:->[batch:h:w:c]
        """
        fig = plt.figure(figsize=(4, 4))

        for i in range(images.shape[0]):
            plt.subplot(nrow,  ncol, i+1)
            plt.axis('off')
            plt.grid(False)
            plt.imshow(images[i, :, :, 0] * 255)# -1～1に正規化されている画像を0～255に変換して表示
            
        return fig

