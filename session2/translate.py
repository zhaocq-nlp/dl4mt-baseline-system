'''
Translates a source file using a translation model.
'''
import argparse

import os
import numpy
import cPickle as pkl
import sys
import theano


from nmt import (build_sampler, gen_sample, load_params,
                 init_params, init_tparams)

from multiprocessing import Process, Queue


def main(model, dictionary, dictionary_target, source_file, saveto, k=5, pkl_file=None,
         normalize=False, output_attention=False):

    # load model model_options
    if pkl_file is None:
        pkl_file = model + '.pkl'
    with open(pkl_file, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f) # word2id
    word_idict = dict() # id2word
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # create input and output queues for processes

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname):
        retval = []
        retval_ori = []
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                words = line.strip().split()
                retval_ori.append(line.strip())
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                x += [0]
                retval.append(x)
        return retval, retval_ori

    print 'Translating ', source_file, '...'
    sys.stdout.flush()

    n_samples, n_samples_src = _send_jobs(source_file)

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # allocate model parameters
    # params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq):
        # sample given an input sequence and obtain scores
        sample, score, att = gen_sample(tparams, f_init, f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   options, trng=trng, k=k, maxlen=200,
                                   stochastic=False, argmax=False)
        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        # return sample[sidx], att[sidx]
        return sample[sidx], numpy.array(att[sidx])

    def _output_attention(sent_idx, att):
        dirname = saveto + '.attention'
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        with open(dirname + '/' + str(sent_idx), 'w') as fp:
            fp.write("%d %d\n" % (att.shape[0], att.shape[1]))
            for row in att:
                fp.write(str(row.argmax()) + " " + ' '.join([str(x) for x in row]) + '\n')

    # translation
    ys = []
    atts = []
    idx = 0

    for x in n_samples:
        y, att = _translate(x)
        ys.append(y)
        atts.append(att)
        print idx
        idx += 1
    trans = _seqs2words(ys)

    # save
    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)
    if output_attention:
        with open(saveto + '.att', 'w') as f:
            for idx, (x, y, att) in enumerate(zip(n_samples_src, trans, atts)):
                print >>f, ('%d ||| %s ||| 0 ||| %s ||| %d %d'
                            % (idx, y, x, att.shape[1], att.shape[0]))
                for hehe in att:
                    print >>f, ' '.join([str(x) for x in hehe])
                print >>f

    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # k: beam size
    parser.add_argument('-k', type=int, default=5)
    # n: if normalize
    parser.add_argument('-n', action="store_true", default=True)
    # if output attention
    parser.add_argument('-a', action="store_true", default=False)
    # pkl model
    parser.add_argument('-m', type=str, default=None)
    # model.npz
    parser.add_argument('model', type=str)
    # source side dictionary
    parser.add_argument('dictionary', type=str)
    # target side dictionary
    parser.add_argument('dictionary_target', type=str)
    # source file
    parser.add_argument('source', type=str)
    # translation file
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    import datetime
    start_time = datetime.datetime.now()

    main(args.model, args.dictionary, args.dictionary_target, args.source,
         args.saveto, k=args.k, pkl_file=args.m, normalize=args.n,
         output_attention=args.a)

    print 'Elapsed Time: %s' % str(datetime.datetime.now() - start_time)
