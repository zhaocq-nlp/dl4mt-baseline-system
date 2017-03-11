import os
import sys
import subprocess

class BleuValidator:

    def __init__(self, options, **kwards):
        self.src_dict = options['dictionaries'][0]
        self.trg_dict = options['dictionaries'][1]
        self.save_freq = options['saveFreq']
        self.tmp_dir = kwards['tmp_dir']
        self.translate_script = kwards['translate_script']
        self.bleu_script = kwards['bleu_script']
        self.valid_src = kwards['valid_src']
        self.valid_trg = kwards['valid_trg']
        self.val_burn_in = int(kwards['val_burn_in'])
        self.valid_freq = int(kwards['valid_freq'])
        self.val_burn_in_fine = int(kwards['val_burn_in_fine'])
        self.valid_freq_fine = int(kwards['valid_freq_fine'])
        self.valid_freq_fine_half = self.valid_freq_fine / 2
        if self.valid_freq_fine_half == 0:
            self.valid_freq_fine_half = 1
        self.valid_freq_final = (int(kwards['valid_freq_final'])
                                 if int(kwards['valid_freq_final']) < self.valid_freq_fine_half
                                 else self.valid_freq_fine_half)

        os.system('mkdir -p %s' % self.tmp_dir)
        self.check_script() # check bleu script

    def to_bleu_cmd(self, trans_file):
        #TODO according to the specific bleu script
        #TODO here is the multi-bleu.pl version
        return '%s %s < %s' % (self.bleu_script, self.valid_trg, trans_file)

    @staticmethod
    def parse_bleu_result(bleu_result):
        '''
        parse bleu result string
        :param bleu_result:
        multi-bleu.perl example:
        BLEU = 33.55, 71.9/43.3/26.1/15.7 (BP=0.998, ratio=0.998, hyp_len=26225, ref_len=26289)
        :return: float(33.55) or -1 for error
        '''
        #TODO according to the specific bleu script
        #TODO here is the multi-bleu.pl version
        bleu_result = bleu_result.strip()
        if bleu_result == '':
            return -1.
        try:
            bleu = float(bleu_result[7:bleu_result.index(',')])
        except ValueError:
            bleu = -1.
        return bleu

    def check_script(self):
        if not os.path.exists(self.bleu_script):
            print 'bleu script not exists: %s' % self.bleu_script
            sys.exit(0)
        if not os.path.exists(self.valid_src):
            print 'valid src file not exists: %s' % self.valid_src
            sys.exit(0)

        if os.path.exists(self.valid_trg):
            cmd = self.to_bleu_cmd(self.valid_trg)
        elif os.path.exists(self.valid_trg + str(0)):
            cmd = self.to_bleu_cmd(self.valid_trg + str(0))
        else:
            print 'valid trg file not exists: %s or %s0' % (self.valid_trg, self.valid_trg)
            sys.exit(0)

        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        popen.wait()
        bleu = BleuValidator.parse_bleu_result(''.join(popen.stdout.readlines()).strip())
        if bleu == -1:
            print 'Fail to run script: %s. Please CHECK' % self.bleu_script
            sys.exit(0)
        print 'Successfully test bleu script: %s' % self.bleu_script

    def check_save_decode(self, uidx):
        '''
        check whether to save model and decode
        :param uidx:
        :return: 1. whether to save model, 2. whether to decode
        '''
        if uidx < self.val_burn_in:
            if uidx % self.save_freq == 0:
                return True, False
            else:
                return False, False
        if uidx <= self.val_burn_in_fine and uidx % self.valid_freq == 0:
            return True, True
        if uidx > self.val_burn_in_fine and uidx <= 2*self.val_burn_in and uidx % self.valid_freq_fine == 0:
            return True, True
        if uidx > 2 * self.val_burn_in and uidx % self.valid_freq_fine_half == 0:
            return True, True
        if uidx > 2 * self.val_burn_in_fine and uidx % self.valid_freq_fine == 0:
            return True, True
        return False, False

    def decode(self, device, trans_saveto, model_file, pkl_file):
        # TODO python translate.py -n xxx xxx xxx
        cmd = "THEANO_FLAGS='device=%s,floatX=float32' python %s -n -m %s %s %s %s %s %s" \
            % (device,
               self.translate_script,
               pkl_file,
               model_file,
               self.src_dict,
               self.trg_dict,
               self.valid_src,
               os.path.join(self.tmp_dir, trans_saveto))
        print 'running: %s' % cmd
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        # return subprocess.Popen(cmd, shell=True)

    def test_bleu(self, trans_file):
        popen = subprocess.Popen(self.to_bleu_cmd(os.path.join(self.tmp_dir, trans_file)),
                                 stdout=subprocess.PIPE, shell=True)
        popen.wait()
        bleu = BleuValidator.parse_bleu_result(''.join(popen.stdout.readlines()).strip())
        if bleu == -1:
            print 'Fail to run script: %s, for testing trans file: %s' % (self.bleu_script, trans_file)
            sys.exit(0)
        return bleu

