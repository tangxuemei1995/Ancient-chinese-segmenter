from seqeval.metrics import f1_score, precision_score, recall_score

def cws_evaluate_word_PRF(y_pred, y):
    #dict = {'E': 2, 'S': 3, 'B':0, 'I':1}
    cor_num = 0
    yp_wordnum = y_pred.count('E')+y_pred.count('S')
    yt_wordnum = y.count('E')+y.count('S')
    start = 0
    for i in range(len(y)):
        if y[i] == 'E' or y[i] == 'S':
            flag = True
            for j in range(start, i+1):
                if y[j] != y_pred[j]:
                    flag = False
            if flag:
                cor_num += 1
            start = i+1

    P = cor_num / float(yp_wordnum) if yp_wordnum > 0 else -1
    R = cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    F = 2 * P * R / (P + R)
    print('P: ', P)
    print('R: ', R)
    print('F: ', F)
    return P, R, F
    
def eval_sentence(y_pred, y, sentence, word2id):
    words = sentence.split(' ')
    seg_true = []
    seg_pred = []
    word_true = ''
    word_pred = ''

    y_word = []
    y_pos = []
    y_pred_word = []
    y_pred_pos = []
    for y_label, y_pred_label in zip(y, y_pred):
        y_word.append(y_label[0])
        y_pos.append(y_label[2:])
        y_pred_word.append(y_pred_label[0])
        y_pred_pos.append(y_pred_label[2:])

    for i in range(len(y_word)):
        word_true += words[i]
        word_pred += words[i]
        if y_word[i] in ['S', 'E']:
            pos_tag_true = y_pos[i]
            word_pos_true = word_true + '_' + pos_tag_true
            if word_true not in word2id:
                word_pos_true = '*' + word_pos_true + '*'
            seg_true.append(word_pos_true)
            word_true = ''
        if y_pred_word[i] in ['S', 'E']:
            pos_tag_pred = y_pred_pos[i]
            word_pos_pred = word_pred + '_' + pos_tag_pred
            seg_pred.append(word_pos_pred)
            word_pred = ''

    seg_true_str = ' '.join(seg_true)
    seg_pred_str = ' '.join(seg_pred)
    return seg_true_str, seg_pred_str


def pos_evaluate_word_PRF(y_pred, y):
    #dict = {'E': 2, 'S': 3, 'B':0, 'I':1}
    y_word = []
    y_pos = []
    y_pred_word = []
    y_pred_pos = []
    for y_label, y_pred_label in zip(y, y_pred):
        y_word.append(y_label[0])
        y_pos.append(y_label[2:])
        y_pred_word.append(y_pred_label[0])
        y_pred_pos.append(y_pred_label[2:])

    word_cor_num = 0
    pos_cor_num = 0
    yp_wordnum = y_pred_word.count('E')+y_pred_word.count('S')
    yt_wordnum = y_word.count('E')+y_word.count('S')
    start = 0
    for i in range(len(y_word)):
        if y_word[i] == 'E' or y_word[i] == 'S':
            word_flag = True
            pos_flag = True
            for j in range(start, i+1):
                if y_word[j] != y_pred_word[j]:
                    word_flag = False
                    pos_flag = False
                    break
                if y_pos[j] != y_pred_pos[j]:
                    pos_flag = False
            if word_flag:
                word_cor_num += 1
            if pos_flag:
                pos_cor_num += 1
            start = i+1

    wP = word_cor_num / float(yp_wordnum) if yp_wordnum > 0 else -1
    wR = word_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    wF = 2 * wP * wR / (wP + wR) if wP + wR > 0 else 0

    pP = pos_cor_num / float(yp_wordnum) if yp_wordnum > 0 else -1
    pR = pos_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    pF = 2 * pP * pR / (pP + pR)
    #
    # pP = precision_score([y], [y_pred])
    # pR = recall_score([y], [y_pred])
    # pF = f1_score([y], [y_pred])

    return (100 * wP, 100 * wR, 100 * wF), (100 * pP, 100 * pR, 100 * pF)
    
def cws_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id):
    cor_num = 0
    yt_wordnum = 0
    for y_pred, y, sentence in zip(y_pred_list, y_list, sentence_list):
        start = 0
        for i in range(len(y)):
            if y[i] == 'E' or y[i] == 'S':
                word = ''.join(sentence[start:i+1])
                if word in word2id:
                    start = i + 1
                    continue
                flag = True
                yt_wordnum += 1
                for j in range(start, i+1):
                    if y[j] != y_pred[j]:
                        flag = False
                if flag:
                    cor_num += 1
                start = i + 1

    OOV = cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    return OOV
