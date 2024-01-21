import tensorflow as tf
from tensorflow.keras import layers



# 使用する正規化層のクラスを返す関数
def get_bn_layer(norm):
    bn_layers = {
        'batch': layers.BatchNormalization,
        'layer': layers.LayerNormalization
    }
    assert norm in bn_layers, f"Unsupported normalization layer {norm}"
    return bn_layers[norm]


class GeneratorBlock(tf.keras.Model):
    """
    ResNetブロックを実装したクラス
    upsampleがTrueの場合2倍の特徴マップ、Falseの場合1倍の特徴マップを返す
    """
    def __init__(self, norm, filter, kernels=[3, 3, 3], name='generator_block', **kwargs):
        super().__init__(name=name, **kwargs)
        upkernel, kernel2, kernel3 = kernels
        
        # 使用する正規化層のクラスを取得
        Norm = get_bn_layer(norm) # -> class

        """ アップサンプリング層"""
        # self.upSamplingLayer = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='upSampling')
        self.upSamplingLayer = layers.Conv2DTranspose(filters=filter, kernel_size=upkernel, strides=2, padding='same', name='conv2DTranspose')

        """ 畳み込み層, バッチ正則化層"""
        self.conv2D_1 = layers.Conv2D(filters=filter, kernel_size=1, strides=1, padding="same", use_bias=False, name='conv_1')
        self.Norm_1 = Norm(name=f'{norm}Norm_1')

        # strides=1かつpadding="same"の場合、出力サイズは入力サイズと同じになる
        self.conv2D_2 = layers.Conv2D(filters=filter, kernel_size=kernel2, strides=1, padding="same", use_bias=False, name='conv_2')
        self.Norm_2 = Norm(name=f'{norm}Norm_2')

        self.conv2D_3 = layers.Conv2D(filters=filter, kernel_size=kernel3, strides=1, padding="same", use_bias=False, name='conv_3')
        self.Norm_3 = Norm(name=f'{norm}Norm_3')

        self.conv2D_4 = layers.Conv2D(filters=filter, kernel_size=1, strides=1, padding="same", use_bias=False, name='conv_4')
        self.Norm_4 = Norm(name=f'{norm}Norm_4')
        
        """ 活性化関数 """
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)

        """ skip connection用のAdd層"""
        self.add = layers.Add()

    # 順伝播処理
    def call(self, x, training, upsample=True):
        # tf.print("inputs:", inputs.shape)
        if upsample:
            x = self.upSamplingLayer(x)

        skip = self.conv2D_1(x)
        skip = self.Norm_1(skip, training=training)
        # tf.print("skip:", skip.shape)

        x = self.conv2D_2(x)
        x = self.Norm_2(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv2D_3(x)
        x = self.Norm_3(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv2D_4(x)
        x = self.add([x, skip])
        x= self.Norm_4(x, training=training)
        x = self.leaky_relu(x)

        return x


class EncoderBlock(tf.keras.Model):
    """
    ResNetブロックを実装したクラス
    poolがTrueの場合1/2倍の特徴マップ、Falseの場合1倍の特徴マップを返す
    """
    def __init__(self, norm, filter, kernels=[3, 3, 3], name='encoder_block', **kwargs):
        super().__init__(name=name, **kwargs)
        poolkernel, kernel2, kernel3 = kernels

        # 使用する正規化層のクラスを取得
        Norm = get_bn_layer(norm) # -> class

        """ 畳み込み層, バッチ正則化層"""
        self.conv2D_1 = layers.Conv2D(filters=filter, kernel_size=1, strides=1, padding="same", use_bias=False, name='conv_1')
        self.Norm_1 = Norm(name=f'{norm}Norm_1')

        # strides=1かつpadding="same"の場合、出力サイズは入力サイズと同じになる
        self.conv2D_2 = layers.Conv2D(filters=filter, kernel_size=kernel2, strides=1, padding="same", use_bias=False, name='conv_2')
        self.Norm_2 = Norm(name=f'{norm}Norm_2') 

        self.conv2D_3 = layers.Conv2D(filters=filter, kernel_size=kernel3, strides=1, padding="same", use_bias=False, name='conv_3')
        self.Norm_3 = Norm(name=f'{norm}Norm_3')

        self.conv2D_4 = layers.Conv2D(filters=filter, kernel_size=1, strides=1, padding="same", use_bias=False, name='conv_4')
        self.Norm_4 = Norm(name=f'{norm}Norm_4')
        
        """ 活性化関数"""
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)

        """ skip connection用のAdd層"""
        self.add = layers.Add()

        """ プーリング層"""
        # self.Pooling = layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding="valid", name='pooling')
        self.Pooling = layers.Conv2D(filters=filter, kernel_size=poolkernel, strides=2, padding="same", use_bias=False, name='conv2D_pooling')  

    # 順伝播処理
    def call(self, x, training, pool=True):
        skip = self.conv2D_1(x)
        skip = self.Norm_1(skip, training=training)

        x = self.conv2D_2(x)
        x = self.Norm_2(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv2D_3(x)
        x = self.Norm_3(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv2D_4(x)
        x = self.add([x, skip]) # skip connection
        x= self.Norm_4(x, training=training)
        x = self.leaky_relu(x)

        if pool:
            x = self.Pooling(x)

        return x
    

class Encoder(tf.keras.Model):
    def __init__(
        self, 
        latent_size,  # 潜在変数zの次元数
        name='encoder',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.leaky_relu = layers.LeakyReLU(alpha=0.2)

        encoder_conv_filters = [4, 8, 16, 32, 64, 128]
        self.encoder_block_1 = EncoderBlock('batch', encoder_conv_filters[0], kernels=[4, 3, 3], name='encoder_block_1')
        self.encoder_block_2 = EncoderBlock('batch', encoder_conv_filters[1], kernels=[4, 3, 3], name='encoder_block_2')
        # self.encoder_block_3 = EncoderBlock('batch', encoder_conv_filters[2], kernels=[4, 3, 3], name='encoder_block_3')
        # self.encoder_block_4 = EncoderBlock('batch', encoder_conv_filters[3], kernels=[4, 3, 3], name='encoder_block_4')
        # self.encoder_block_5 = EncoderBlock('batch', encoder_conv_filters[4], kernels=[4, 3, 3], name='encoder_block_5')
        # self.encoder_block_6 = EncoderBlock('batch', encoder_conv_filters[5], kernels=[4, 3, 3], name='encoder_block_6')

        self.conv1 = layers.Conv2D(filters=encoder_conv_filters[2], kernel_size=4, strides=2, padding="same", use_bias=False, name='conv1')
        self.Norm1 = layers.BatchNormalization(name='encoder_batchNorm1')
        self.conv2 = layers.Conv2D(filters=encoder_conv_filters[3], kernel_size=4, strides=2, padding="same", use_bias=False, name='conv2')
        self.Norm2 = layers.BatchNormalization(name='encoder_batchNorm2')
        self.conv3 = layers.Conv2D(filters=encoder_conv_filters[4], kernel_size=4, strides=2, padding="same", use_bias=False, name='conv3')
        self.Norm3 = layers.BatchNormalization(name='encoder_batchNorm3')
        self.conv4 = layers.Conv2D(filters=encoder_conv_filters[5], kernel_size=4, strides=2, padding="same", use_bias=False, name='conv4')
        self.Norm4 = layers.BatchNormalization(name='encoder_batchNorm4')

        # conv2Dでpoolingするときは、padding="same"すると、stride=nのときに出力サイズが1/nになる
        # self.Pooling = layers.Conv2D(filters=encoder_conv_filters[-1], kernel_size=4, strides=3, padding="same", use_bias=False, name='conv2D_pooling')  

        """global average pooling"""
        self.conv_last=layers.Conv2D(
                filters=latent_size,
                kernel_size=1,
                strides=1,
                padding="valid",
                use_bias=False,
                name = 'encoder_conv_last'
        )
        self.global_average_layer = layers.GlobalAveragePooling2D() # => TensorShape([None, latent_size])        
        
    def get_functionalModel(self, input_shape):
        """
        summary()などbuild済みモデルに対してのみ呼び出せるメソッドを使えるようにするためにfunctionalModelに変換したモデルを返す
        ======================================================================================
        input_shape:バッチサイズを除いた入力データのサイズ。例：縦横28px,RGB画像(28, 28, 3)

        注)本メソッド経由でcallを呼び出すと、入力xがKerasTensorになっているため、
        tf.print(tf.shape(x))の部分で以下のエラーが発生する。代わりにx.shapeを使えば表示できる。
        "Cannot convert a symbolic Keras"
        なお、表示させない場合(shape=tf.shape(x)など)は問題ない。
        ---------------------------------------------------------------------------------------
        return:
        functionalAPIに変換したモデルオブジェクト
        """
        x = layers.Input(shape=input_shape, name='layer_in')
        functionalModel = tf.keras.Model(
            inputs=[x],
            outputs=self.call(x, training=False),
            name="functionalModel"
        )
        
        return functionalModel
        
    # 順伝播処理
    def call(self, inputs, training):
        x = self.encoder_block_1(inputs, training=training)
        x = self.encoder_block_2(x, training=training)
        # x = self.encoder_block_3(x, training=training)
        # x = self.encoder_block_4(x, training=training)
        # x = self.encoder_block_5(x, training=training)
        # x = self.encoder_block_6(x, training=training)

        x = self.conv1(x)
        x = self.Norm1(x, training=training)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.Norm2(x, training=training)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.Norm3(x, training=training)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.Norm4(x, training=training)
        x = self.leaky_relu(x)
        
        # x = self.Pooling(x)
        # x = self.leaky_relu(x)

        x = self.conv_last(x)
        x = self.global_average_layer(x)
        
        return x


class Generator(tf.keras.Model):
    def __init__(
        self, 
        latent_size,  # 潜在変数zの次元数
        channels=3, # 出力画像のチャンネル数
        name='generator',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.reshape_input = layers.Reshape((1, 1, latent_size), name='generator_reshape_1')
        
        generator_conv_filters = [256, 256, 128, 128, 64, 32]
        self.convTrans1 = layers.Conv2DTranspose(filters=generator_conv_filters[0], kernel_size=2, strides=1, padding='valid', name='conv2DTranspose1')
        self.Norm1 = layers.BatchNormalization(name='generator_batchNorm1')

        self.generator_block_1 = GeneratorBlock('batch', generator_conv_filters[1], kernels=[4, 3, 3], name='generator_block_1')
        self.generator_block_2 = GeneratorBlock('batch', generator_conv_filters[2], kernels=[4, 3, 3], name='generator_block_2')
        # self.generator_block_3 = GeneratorBlock('batch', generator_conv_filters[3], kernels=[4, 3, 3], name='generator_block_3')
        # self.generator_block_4 = GeneratorBlock('batch', generator_conv_filters[4], kernels=[4, 3, 3], name='generator_block_4')
        # self.generator_block_5 = GeneratorBlock('batch', generator_conv_filters[5], kernels=[4, 3, 3], name='generator_block_5')

        self.convTrans2 = layers.Conv2DTranspose(filters=generator_conv_filters[3], kernel_size=4, strides=2, padding='same', name='conv2DTranspose2')
        self.Norm2 = layers.BatchNormalization(name='generator_batchNorm2')
        self.convTrans3 = layers.Conv2DTranspose(filters=generator_conv_filters[4], kernel_size=4, strides=2, padding='same', name='conv2DTranspose3')
        self.Norm3 = layers.BatchNormalization(name='generator_batchNorm3')
        self.convTrans4 = layers.Conv2DTranspose(filters=generator_conv_filters[5], kernel_size=4, strides=2, padding='same', name='conv2DTranspose4')
        self.Norm4 = layers.BatchNormalization(name='generator_batchNorm4')

        self.lastLayer_Conv2D = layers.Conv2D(
                filters=channels,
                kernel_size=1,
                strides=1,
                padding="valid",
                use_bias=False,
                name = 'generator_conv_last'
            )
        
        self.dropout = layers.SpatialDropout2D(rate=0.5, name='generator_dropout')
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)
        self.tanh = tf.keras.layers.Activation('tanh')
        
    def get_functionalModel(self, input_shape):
        """
        summary()などbuild済みモデルに対してのみ呼び出せるメソッドを使えるようにするためにfunctionalModelに変換したモデルを返す
        ======================================================================================
        input_shape:バッチサイズを除いた入力データのサイズ。例：縦横28px,RGB画像(28, 28, 3)

        注)本メソッド経由でcallを呼び出すと、入力xがKerasTensorになっているため、
        tf.print(tf.shape(x))の部分で以下のエラーが発生する。代わりにx.shapeを使えば表示できる。
        "Cannot convert a symbolic Keras"
        なお、表示させない場合(shape=tf.shape(x)など)は問題ない。
        ---------------------------------------------------------------------------------------
        return:
        functionalAPIに変換したモデルオブジェクト
        """
        x = layers.Input(shape=input_shape, name='layer_in')
        functionalModel = tf.keras.Model(
            inputs=[x],
            outputs=self.call(x, training=False),
            name="functionalModel"
        )
        
        return functionalModel
        
    # 順伝播処理
    def call(self, x, training):
        x = self.reshape_input(x) # => TensorShape([None, 2, 2, filters)
        x = self.convTrans1(x)
        x = self.Norm1(x, training=training)
        x = self.leaky_relu(x)

        x = self.generator_block_1(x, training=training)
        x = self.dropout(x, training=training)
        x = self.generator_block_2(x, training=training)
        x = self.dropout(x, training=training)
        # x = self.generator_block_3(x, training=training)
        # x = self.dropout(x, training=training)
        # x = self.generator_block_4(x, training=training)
        # x = self.dropout(x, training=training)
        # x = self.generator_block_5(x, training=training)

        x = self.convTrans2(x)
        x = self.Norm2(x, training=training)
        x = self.leaky_relu(x)
        x = self.convTrans3(x)
        x = self.Norm3(x, training=training)
        x = self.leaky_relu(x)
        x = self.convTrans4(x)
        x = self.Norm4(x, training=training)
        x = self.leaky_relu(x)

        x = self.lastLayer_Conv2D(x)
        x = self.tanh(x)
        
        return x
    

class Discriminator(tf.keras.Model):
    def __init__(
        self, 
        latent_size,  # 潜在変数zの次元数
        name='discriminator',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.leaky_relu = layers.LeakyReLU(alpha=0.1)
        self.dropout = layers.SpatialDropout2D(rate=0.3, name='discriminator_dropout')

        """latent layer"""
        self.reshape_latent = layers.Reshape((1, 1, latent_size), name='discriminator_reshape_latent')
        self.conv_latent = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='valid', name = 'discriminator_conv_latent')
        
        """ image layer """
        discriminator_conv_filters = [8, 16, 32, 64, 64, 128]
        self.discriminator_block_1 = EncoderBlock('layer', discriminator_conv_filters[0], kernels=[4, 3, 3], name='discriminator_block_1')
        self.discriminator_block_2 = EncoderBlock('layer', discriminator_conv_filters[1], kernels=[4, 3, 3], name='discriminator_block_2')
        # self.discriminator_block_3 = EncoderBlock('layer', discriminator_conv_filters[2], kernels=[4, 3, 3], name='discriminator_block_3')
        # self.discriminator_block_4 = EncoderBlock('layer', discriminator_conv_filters[3], kernels=[4, 3, 3], name='discriminator_block_4')
        # self.discriminator_block_5 = EncoderBlock('layer', discriminator_conv_filters[4], kernels=[4, 3, 3], name='discriminator_block_5')

        self.conv1 = layers.Conv2D(filters=discriminator_conv_filters[2], kernel_size=4, strides=2, padding="same", use_bias=False, name='conv1')
        self.Norm1 = layers.BatchNormalization(name='encoder_batchNorm1')
        self.conv2 = layers.Conv2D(filters=discriminator_conv_filters[3], kernel_size=4, strides=2, padding="same", use_bias=False, name='conv2')
        self.Norm2 = layers.BatchNormalization(name='encoder_batchNorm2')
        self.conv3 = layers.Conv2D(filters=discriminator_conv_filters[4], kernel_size=4, strides=2, padding="same", use_bias=False, name='conv3')
        self.Norm3 = layers.BatchNormalization(name='encoder_batchNorm3')
        self.conv4 = layers.Conv2D(filters=discriminator_conv_filters[5], kernel_size=4, strides=2, padding="same", use_bias=False, name='conv4')
        self.Norm4 = layers.BatchNormalization(name='encoder_batchNorm4')
        
        # conv2Dでpoolingするときは、padding="same"すると、stride=nのときに出力サイズが1/nになる
        # self.Pooling = layers.Conv2D(filters=discriminator_conv_filters[-1], kernel_size=4, strides=2, padding="same", use_bias=False, name='conv2D_pooling')          

        """ concated layer """    
        self.concat = layers.Concatenate(axis=-1, name='discriminator_concat')
        self.conv1_concated = layers.Conv2D(filters=512,  kernel_size=1, strides=1, padding='valid', name = 'discriminator_conv1_concated')
        self.conv2_concated = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='valid', name = 'discriminator_conv2_concated')

    def get_functionalModel(self, input_shape):
        """
        summary()などbuild済みモデルに対してのみ呼び出せるメソッドを使えるようにするためにfunctionalModelに変換したモデルを返す
        ======================================================================================
        input_shape:バッチサイズを除いた入力データのサイズ。例：縦横28px,RGB画像(28, 28, 3)

        注)本メソッド経由でcallを呼び出すと、入力xがKerasTensorになっているため、
        tf.print(tf.shape(x))の部分で以下のエラーが発生する。代わりにx.shapeを使えば表示できる。
        "Cannot convert a symbolic Keras"
        なお、表示させない場合(shape=tf.shape(x)など)は問題ない。
        ---------------------------------------------------------------------------------------
        return:
        functionalAPIに変換したモデルオブジェクト
        """
        x = layers.Input(shape=input_shape[0], name='layer_in_x')
        z = layers.Input(shape=input_shape[1], name='layer_in_z')
        functionalModel = tf.keras.Model(
            inputs=[x, z],
            outputs=self.call(x, z, training=False),
            name="functionalModel"
        )
        
        return functionalModel
        
    # 順伝播処理
    def call(self, x, z, training):
        """ latent path """
        l = self.reshape_latent(z)
        l = self.conv_latent(l)
        l = self.leaky_relu(l)
        l = self.dropout(l, training=training)

        """ image path """
        x = self.discriminator_block_1(x, training=training)
        x = self.dropout(x, training=training)
        x = self.discriminator_block_2(x, training=training)
        x = self.dropout(x, training=training)
        # x = self.discriminator_block_3(x, training=training)
        # x = self.dropout(x, training=training)
        # x = self.discriminator_block_4(x, training=training)
        # x = self.dropout(x, training=training)
        # x = self.discriminator_block_5(x, training=training)
        # x = self.dropout(x, training=training)

        x = self.conv1(x)
        x = self.Norm1(x, training=training)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.Norm2(x, training=training)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.Norm3(x, training=training)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.Norm4(x, training=training)
        x = self.leaky_relu(x)

        # x = self.Pooling(x)
        # x = self.leaky_relu(x)
        # x = self.dropout(x, training=training)

        """ concated path """
        x = self.concat([x, l])
        x = self.conv1_concated(x)
        x = self.leaky_relu(x)
        x = self.dropout(x, training=training)

        feature = tf.reshape(x, [tf.shape(x)[0], -1]) # Discrimination Lossを測るためにこの値も返す。
        x = self.conv2_concated(x)

        return tf.squeeze(x), feature