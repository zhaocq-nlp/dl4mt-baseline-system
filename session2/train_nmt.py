import numpy
import os
import sys
from nmt import train


def main(job_id, params):
    # print params

    # recommendation:
    #   val_burn_in: 100000
    #   valid_freq: 10000
    #   val_burn_in_fine: 150000
    #   valid_freq_fine: 4000
    #   valid_freq_final: 1500
    # save always with decode (see bleu_validator.py in detail)
    #   only save model while uidx < $val_burn_in
    #   save and decode each $valid_freq updates while uidx <= $val_burn_in_fine
    #   save and decode each $valid_freq_fine while $val_burn_in_fine < uidx <= 2 * $val_burn_in
    #   save and decode each $valid_freq_fine/2 while uidx > 2 * $val_burn_in
    #   save and decode each $valid_freq_final while uidx > 2 * $val_burn_in_fine
    bleuvalid_params = {
     'tmp_dir': './tmp_trans/',
     'translate_script': './translate.py',
     'bleu_script': './multi-bleu.perl',
     'valid_src': '../data/MT02/ch',
     'valid_trg': '../data/MT02/en',
     'val_burn_in': 100000,
     'valid_freq': 10000,
     'val_burn_in_fine': 150000,
     'valid_freq_fine': 4000,
     'valid_freq_final': 1500,
    }

    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     maxlen=50,
                     batch_size=80,
                     valid_batch_size=80,
                     validFreq=100,
                     dispFreq=10,
                     saveFreq=20000,
                     sampleFreq=100,
                     max_epochs=50,  # max iteration
                     patience=70,  # early stop patience with BLEU score
                     finish_after=1000000,  # max updates
                     datasets=['../data/125w/zh-en.zh.utf8',
                               '../data/125w/zh-en.en.utf8'],
                     valid_datasets=['../data/MT02/ch',
                                     '../data/MT02/en0'],
                     dictionaries=['../data/125w/zh-en.zh.utf8.pkl',
                                   '../data/125w/zh-en.en.utf8.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False,
                     **bleuvalid_params)

    # for debug
    '''
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=10,
                     dim=5,
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=10,
                     maxlen=50,
                     batch_size=2,
                     valid_batch_size=32,
                     validFreq=2,
                     dispFreq=1,
                     saveFreq=2,
                     sampleFreq=3,
                     max_epochs=600,
                     finish_after=30,
                     datasets=['../../test_data/zh.test.ori',
                               '../../test_data/en.test.ori'],
                     valid_datasets=['../../test_data/zh.test.ori',
                                     '../../test_data/en.test.ori'],
                     # dictionaries=['../data/hal/train/tok/en.pkl',
                     #               '../data/hal/train/tok/fr.pkl'],
                     dictionaries=['../../test_data/zh.test.pkl',
                                   '../../test_data/en.test.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False,
                     **bleuvalid_params)
    '''
    
    return validerr

if __name__ == '__main__':

    import datetime
    start_time = datetime.datetime.now()
    main(0, {
        'model': ['model_hal.npz'],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [True]})

    print 'Elapsed Time: %s' % str(datetime.datetime.now() - start_time)

