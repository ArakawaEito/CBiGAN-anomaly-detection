import numpy as np
import librosa as lb

'''親クラスとして指定した確率から，
データごとに処理をするかしないかを決めるTransformWaveformクラス.
全ての音声に同じ処理をせず，ランダムにさまざまな特徴のデータを作成する．
'''
# 確率的に処理を施すクラス
class TransformWaveform:
    def __init__(self, always_apply=False, prob=0.5):
        self.always_apply = always_apply
        self.prob = prob

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.prob:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError

'''複数の処理のリストを引数として入力し、一つの関数にするためのクラス.
一つにまとめることにより，様々な音声処理を抜いたり加えたりカスタマイズしやすくなる

'''
# 複数の処理をまとめるクラス
class Multiple:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y

'''
Resamplingはメモリの省力化や精度の向上のために、
音声ファイルのサンプリング周波数を統一するための処理．
'''
# リサンプリング
class ResampleWaveform(TransformWaveform):   
    def __init__(self, sr, resample_sr, always_apply=True, prob=0.5):
        super().__init__(always_apply, prob)
        self.sr = sr #元のサンプリング周波数
        self.resample_sr =resample_sr #リサンプリング後のサンプリング周波数

    def apply(self, y: np.ndarray, **params):
        y_resampled = lb.resample(y, self.sr, self.resample_sr) # 新たなサンプリング周波数としてresample_srを指定
        return y_resampled

# ホワイトノイズ
class GaussianNoiseSNR(TransformWaveform):
    def __init__(self, always_apply=True, prob=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, prob)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr) 
        a_signal = np.sqrt(y ** 2).max() 
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max() 
        # ↓右と同じ式(y + white_noise * (a_noise/ a_white) ).astype(y.dtype) 
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented

# ピンクノイズ
import colorednoise as cn
class PinkNoiseSNR(TransformWaveform):
    def __init__(self, always_apply=False, prob=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, prob)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        beta = 1 # （1にするとピンクノイズになる）
        samples = len(y) 
        pink_noise = cn.powerlaw_psd_gaussian(beta, samples)
        a_pink = np.sqrt(pink_noise ** 2).max()
        # ↓右と同じ式(y + pink_noise * (a_noise/ a_pink) ).astype(y.dtype) 
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented

'''PitchShiftは音のピッチ(高低)に関する調整を施すData Augmentationで、
効果として聞こえる音が高く/低くなる．
メルスペクトログラム上では、パターンのある周波数帯が上または下にズレる．
'''
# ピッチシフト
class PitchShift(TransformWaveform):
    def __init__(self, always_apply=False, prob=0.5, max_steps=5, sr=16000):
        super().__init__(always_apply, prob)

        self.max_steps = max_steps
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_steps, self.max_steps) # -self.max_steps以上self.max_steps未満の整数の(一様分布)乱数
        augmented = lb.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps) # n_steps:どのくらい周波数をシフトさせるか
        return augmented

'''TimeShiftは時間をずらす方法．
ずらし方は、前の部分に何も音のないデータを付け加え後ろの部分をカットするか、
前の部分にカットした後ろの部分を付け加えるかの二通り．
'''
# タイムシフト
class TimeShift(TransformWaveform):
    def __init__(self, always_apply=False, prob=0.5, max_shift_second=2, sr=32000, padding_mode="replace"):
        super().__init__(always_apply, prob)
        
        assert padding_mode in ["replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'" 
        self.max_shift_second = max_shift_second
        self.sr = sr
        self.padding_mode = padding_mode

    def apply(self, y: np.ndarray, **params):
        shift = np.random.randint(-self.sr * self.max_shift_second, self.sr * self.max_shift_second)
        augmented = np.roll(y, shift) #np.roll():shift(>0)の分だけ右にシフトする．はみ出た分は後ろに付け加える
        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift] = 0
            else:
                augmented[shift:] = 0
        return augmented

'''VolumeControlは音量を調節する方法.
sine曲線やcosine曲線に合わせて音量を経過時間によって変化させることで
メルスペクトログラムに大きな変化をもたらし様々な特徴を捉えやすくなる．
'''
#　音量
class VolumeShift(TransformWaveform):
    def __init__(self, always_apply=False, prob=0.5, db_limit=10, mode="cosine"):
        super().__init__(always_apply, prob)
        
        assert mode in ["uniform", "fade", "fade", "cosine", "sine"], \
            "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit= db_limit
        self.mode = mode

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.db_limit, self.db_limit)       
        
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1) 
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2) #0～2Πまでのcos値の配列
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2) #0～2Πまでのsin値の配列
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented