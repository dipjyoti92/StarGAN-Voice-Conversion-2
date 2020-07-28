import argparse
from model import Generator
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
import librosa
from utils import *
import glob

# Below is the accent info for the used 70 speakers.

speakers = ['p225', 'p227', 'p228', 'p229', 'p230', 'p231', 'p232', 'p233', 'p234', 'p236', 'p237','p238', 'p239', 'p240', 'p241', 'p243', 'p244', 'p245', 'p246', 'p247', 'p248', 'p249', 'p250', 'p251', 'p252', 'p253', 'p254', 'p255', 'p256', 'p257', 'p258', 'p259', 'p260', 'p261', 'p262', 'p263', 'p264', 'p265', 'p266', 'p267', 'p268', 'p269', 'p270', 'p271', 'p272', 'p273', 'p274', 'p275', 'p276', 'p277', 'p278', 'p279', 'p280', 'p281', 'p282', 'p283', 'p284', 'p285', 'p286', 'p287', 'p288', 'p292', 'p293', 'p294', 'p295', 'p297', 'p298', 'p299', 'p300', 'p360'] 
spk2idx = dict(zip(speakers, range(len(speakers))))

class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, config):
        assert config.trg_spk in speakers, f'The trg_spk should be chosen from {speakers}, but you choose {trg_spk}.'
        # Source speaker
        self.src_spk = config.src_spk
        self.trg_spk = config.trg_spk

        self.mc_files = sorted(glob.glob(join(config.test_data_dir, f'{config.src_spk}*.npy')))
        self.src_spk_stats = np.load(join(config.train_data_dir, f'{config.src_spk}_stats.npz'))
        self.src_wav_dir = f'{config.wav_dir}/{config.src_spk}'

        
        self.trg_spk_stats = np.load(join(config.train_data_dir, f'{config.trg_spk}_stats.npz'))

        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']
        
        self.spk_idx = spk2idx[self.trg_spk]
        spk_cat = to_categorical([self.spk_idx], num_classes=len(speakers))
        self.spk_c_trg = spk_cat
        self.spk_idx_org = spk2idx[self.src_spk]
        spk_cat_org = to_categorical([self.spk_idx_org], num_classes=len(speakers))
        self.spk_c_org = spk_cat_org


    def get_batch_test_data(self, batch_size=4):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data 


def load_wav(wavfile, sr=16000):
    wav, _ = librosa.load(wavfile, sr=sr, mono=True)
    return wav_padding(wav, sr=sr, frame_period=5, multiple = 4)  # TODO
    # return wav

def test(config):
    os.makedirs(join(config.convert_dir, str(config.resume_iters)), exist_ok=True)
    sampling_rate, num_mcep, frame_period=16000, 36, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator().to(device)
    test_loader = TestDataset(config)
    # Restore model
    print(f'Loading the trained models from step {config.resume_iters}...')
    G_path = join(config.model_save_dir, f'{config.resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    # Read a batch of testdata
    test_wavfiles = test_loader.get_batch_test_data(batch_size=config.num_converted_wavs)
    test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in test_wavfiles]

    with torch.no_grad():
        for idx, wav in enumerate(test_wavs):
            print(len(wav))
            wav_name = basename(test_wavfiles[idx])
            # print(wav_name)
            f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = pitch_conversion(f0=f0, 
                mean_log_src=test_loader.logf0s_mean_src, std_log_src=test_loader.logf0s_std_src, 
                mean_log_target=test_loader.logf0s_mean_trg, std_log_target=test_loader.logf0s_std_trg)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
            print("Before being fed into G: ", coded_sp.shape)
            coded_sp_norm = (coded_sp - test_loader.mcep_mean_src) / test_loader.mcep_std_src
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)
            conds_trg = torch.FloatTensor(test_loader.spk_c_trg).to(device)
            conds_org = torch.FloatTensor(test_loader.spk_c_org).to(device)
            # print(spk_conds.size())
            coded_sp_converted_norm = G(coded_sp_norm_tensor, conds_org, conds_trg).data.cpu().numpy()
            coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * test_loader.mcep_std_trg + test_loader.mcep_mean_trg
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            print("After being fed into G: ", coded_sp_converted.shape)
            wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted, 
                                                    ap=ap, fs=sampling_rate, frame_period=frame_period)
            wav_id = wav_name.split('.')[0]
            librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters),
                f'{wav_id}-vcto-{test_loader.trg_spk}.wav'), wav_transformed, sampling_rate)
            if [True, False][0]:
                wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                                ap=ap, fs=sampling_rate, frame_period=frame_period)
                librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters), f'cpsyn-{wav_name}'), wav_cpsyn, sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--num_speakers', type=int, default=70, help='dimension of speaker labels')
    parser.add_argument('--num_converted_wavs', type=int, default=5, help='number of wavs to convert.')
    parser.add_argument('--resume_iters', type=int, default=None, help='step to resume for testing.')
    parser.add_argument('--src_spk', type=str, default='bdl', help = 'target speaker.')
    parser.add_argument('--trg_spk', type=str, default='slt', help = 'target speaker.')

    # Directories.
    parser.add_argument('--train_data_dir', type=str, default='./data/mc/train')
    parser.add_argument('--test_data_dir', type=str, default='./data/mc/test')
    parser.add_argument('--wav_dir', type=str, default="./data/vctk_16")
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--convert_dir', type=str, default='./converted')


    config = parser.parse_args()
    
    print(config)
    if config.resume_iters is None:
        raise RuntimeError("Please specify the step number for resuming.")
    test(config)
