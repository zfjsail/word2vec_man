
import numpy as np
import math
import random


def cos_similarity(x1, x2):
    inner_product = np.dot(x1, x2)
    x1_l2 = np.linalg.norm(x1, 2)
    x2_l2 = np.linalg.norm(x2, 2)
    return inner_product / (x1_l2 * x2_l2)

if __name__ == '__main__':
    fin = open('demo.txt', 'r')
    data = fin.read().split()
    fin.close()
    dic = dict()
    min_reduce = 5  # preserving all words doesn't improve results
    for word in data:
        if word in dic.keys():
            dic[word] += 1
        else:
            dic[word] = 1
    for key in dic.keys():
        if dic[key] < min_reduce:
            dic.pop(key)
    vocab_size = len(dic.keys())
    i = 0
    vocab = []
    for key in dic.keys():
        dic[key] = i
        vocab.append(key)
        i += 1

    window = 5
    negative = 5
    alpha = 0.025
    total_iter = 5
    cur_iter = 0
    losses = [float(0)] * total_iter
    cnt_f = 0
    sentence_position = 0
    word_count = len(data)
    layer1_size = 100
    syn0 = (np.random.rand(vocab_size, layer1_size) - 0.5) / layer1_size  # centralize
    syn1neg = np.zeros((vocab_size, layer1_size), dtype=float)

    EXP_TABLE_SIZE = 1000
    MAX_EXP = 6
    expTable = [0.0] * EXP_TABLE_SIZE  # accelerate obviously!
    for i in range(EXP_TABLE_SIZE):
        expTable[i] = math.exp((i / float(EXP_TABLE_SIZE) * 2 - 1) * MAX_EXP)
        expTable[i] /= (expTable[i] + 1)

    while True:
        if sentence_position % 1000 == 0:
            print sentence_position
        if sentence_position >= word_count:
            losses[cur_iter] /= cnt_f
            cnt_f = 0
            cur_iter += 1
            sentence_position = 0
            print "-------------------"
            if cur_iter >= total_iter:
                break
        word = data[sentence_position]
        if word not in dic.keys():
            sentence_position += 1
            continue
        cur_vocab_index = dic[word]
        neu1 = np.zeros((1, layer1_size))
        neu1e = np.zeros((1, layer1_size))

        cw = 0
        for a in range(0, window * 2 + 1):
            if a != window:
                c = sentence_position - window + a
                if c < 0 or c >= word_count:
                    continue
                last_word = data[c]
                if last_word in dic.keys():
                    last_word_index = dic[last_word]
                else:
                    continue
                neu1 += syn0[last_word_index, :]
                cw += 1
        if cw:
            neu1 /= cw
            # out -> hidden
            for d in range(negative + 1):
                if d == 0:
                    target = cur_vocab_index
                    label = 1
                else:
                    target = random.randint(0, vocab_size - 1)
                    if target == cur_vocab_index:
                        continue
                    label = 0
                f = np.dot(neu1, syn1neg[target, :])
                if f > MAX_EXP:
                    sigmoid_f = 1
                    g = (label - 1) * alpha
                elif f < MAX_EXP:
                    sigmoid_f = 0
                    g = (label - 0) * alpha
                else:
                    sigmoid_f = expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                    g = (label - sigmoid_f) * alpha
                if d == 0:
                    if sigmoid_f:
                        losses[cur_iter] += math.log(sigmoid_f)
                    else:
                        losses[cur_iter] += -6
                else:
                    if sigmoid_f != 1:
                        losses[cur_iter] += math.log(1 - sigmoid_f)
                    else:
                        losses[cur_iter] += -6
                cnt_f += 1
                neu1e += g * syn1neg[target, :]
                syn1neg[target, :] = syn1neg[target, :] + g * neu1

            # hidden -> in
            for a in range(0, window * 2 + 1):
                if a != window:
                    c = sentence_position - window + a
                    if c < 0 or c >= word_count:
                        continue
                    last_word = data[c]
                    if last_word in dic.keys():
                        last_word_index = dic[last_word]
                    else:
                        continue
                    syn0[last_word_index, :] = syn0[last_word_index, :] + neu1e

        sentence_position += 1

    queries = ['music', 'simple', 'people', 'zero', 'much', 'common', 'learn']

    for query in queries:
        print query
        print '----------'
        if query in dic.keys():
            query_index = dic[query]
        else:
            continue
        query_vec = syn0[query_index, :]
        similarity_result = np.zeros(vocab_size)
        for i in range(vocab_size):
            similarity_result[i] = cos_similarity(syn0[i], query_vec)
        sorted_indices = np.argsort(similarity_result)
        for i in range(vocab_size - 2, vocab_size - 12, -1):
            print vocab[sorted_indices[i]]
        print '----------'

    print losses
