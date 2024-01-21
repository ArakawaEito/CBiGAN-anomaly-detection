![Static Badge](https://img.shields.io/badge/python-3.7-blue)
![Static Badge](https://img.shields.io/badge/tensorflow-2.4-FF6F00)
![Static Badge](https://img.shields.io/badge/librosa-0.9.2-4D02A2)
![Static Badge](https://img.shields.io/badge/numpy-1.19.5-013243)
![Static Badge](https://img.shields.io/badge/pandas-1.3.5-150458)

# Anomaly Detection with Consistency Bidirectional GAN (CBiGAN)
 CBiGAN is the model that adds an Encoder to the standard GAN framework. A Generator that can output clear images, and an Encoder that maps images to the latent representations can be trained at the same time.
 <br>
 In detecting anomalies, Reconstruction Error-based method is used, just like Autoencoders. The underlying idea is based on the assumption that if a model can learn a function that compresses and reconstructs normal data, then it will fail to do so when encountered with anomalous data because its function was only trained on normal data. The failure to reconstruct data or, more accurately, the range of the reconstruction error that it entails, can therefore signal the presence of anomalous data.

# Overview
This jupyter notebook explains the results of anomaly detection for sound data with CBiGAN