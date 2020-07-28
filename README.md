# StarGAN-Voice-Conversion-2

This is a pytorch implementation of the paper: [StarGAN-VC2:Rethinking Conditional Methods for StarGAN-Based Voice Conversion](https://arxiv.org/pdf/1907.12279.pdf).

**The converted voice examples are in *converted directory*. The sound quality will improve if more training epochs (~200000) are done.

**VCTK database has been used to train the model with 70 speakers. The convereted samples are a bit noisy because of VCTK data but it can be improved if other clean databases are used.

**Omitted PS in Generator network.

## [Dependencies]
- Python 3.5+
- pytorch 0.4.0+
- librosa 
- pyworld 
- tensorboardX
- scikit-learn
- tqdm


## [Usage]

### Download dataset

Download and unzip [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) corpus to designated directories.

```bash
mkdir ./data
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
unzip VCTK-Corpus.zip -d ./data
```
If the downloaded VCTK is in tar.gz, run this:

```bash
tar -xzvf VCTK-Corpus.tar.gz -C ./data
```
The data directory now looks like this:

```
data
├── vctk
│   ├── p225
│   ├── p226
│   ├── ...
│   └── p360

```

### Preprocess

Extract features (mcep, f0, ap) from each speech clip.  The features are stored as npy files. We also calculate the statistical characteristics for each speaker.

```
python preprocess.py
```

This process may take minutes !

The data directory now looks like this:

```
data
├── vctk (48kHz data)
│   ├── p225
│   ├── p226
│   ├── ...
│   └── p360 
├── vctk_16 (16kHz data)
│   ├── p225
│   ├── p226
│   ├── ...
│   └── p360
├── mc
│   ├── train
│   ├── test


```

### Train

```
python main.py
```


### Convert


```
convert.py --src_spk p262 --trg_spk p272 --resume_iters 210000
```



## [Network structure]

![network](https://github.com/dipjyoti92/StarGAN-Voice-Conversion-2/blob/master/network.png)


## [Reference]

* [StarGAN-VC2 paper](https://arxiv.org/pdf/1907.12279)

* [StarGAN paper](https://arxiv.org/abs/1711.09020)

* [CycleGAN-VC paper](https://arxiv.org/abs/1711.11293)


## [Acknowlegements]

https://github.com/liusongxiang/StarGAN-Voice-Conversion

https://github.com/SamuelBroughton/StarGAN-Voice-Conversion-2





