# -*- coding: utf-8 -*-
import json
import os
import sys
import time
from optparse import OptionParser

from dynet import *
from evaluation import *
from raw_data import make_data_instance
from semafor_evaluation import convert_conll_to_frame_elements


def common_load():
    train_conll = TRAIN_FTE
    train_examples, _, _ = read_conll(train_conll)
    # combined_train = combine_examples(train_examples) #useless
    find_multitokentargets(train_examples, "train")

    # Need to read all LUs before locking the dictionaries.
    target_lu_map, lu_names = create_target_lu_map()
    post_train_lock_dicts()

    lufrmmap, relatedlus = read_related_lus()

    # Read pretrained word embeddings.
    pretrained_map = get_wvec_map()
    PRETRAINED_DIM = len(pretrained_map.values()[0])

    lock_dicts()
    UNKTOKEN = VOCDICT.getid(UNK)

    return {
        'pretrained_map': pretrained_map,
        'PRETRAINED_DIM': PRETRAINED_DIM,
        'target_lu_map': target_lu_map,
        'lu_names': lu_names,
        'lufrmmap': lufrmmap,
        'relatedlus': relatedlus
    }


def combine_examples(corpus_ex):
    """
    Target ID needs to be trained for all targets in the sentence jointly, as opposed to
    frame and arg ID. Returns all target annotations for a given sentence.
    """
    combined_ex = [corpus_ex[0]]
    for ex in corpus_ex[1:]:
        if ex.sent_num == combined_ex[-1].sent_num:
            current_sent = combined_ex.pop()
            target_frame_dict = current_sent.targetframedict.copy()
            target_frame_dict.update(ex.targetframedict)
            current_sent.targetframedict = target_frame_dict
            combined_ex.append(current_sent)
            continue
        combined_ex.append(ex)
    sys.stderr.write("Combined {} instances in data into {} instances.\n".format(
        len(corpus_ex), len(combined_ex)))
    return combined_ex


def print_data_status(fsp_dict, vocab_str):
    sys.stderr.write("# {} = {}\n\tUnseen in dev/test = {}\n\tUnlearnt in dev/test = {}\n".format(
        vocab_str, fsp_dict.size(), fsp_dict.num_unks()[0], fsp_dict.num_unks()[1]))


def get_fn_pos_by_rules(pos, token):
    """
    Rules for mapping NLTK part of speech tags into FrameNet tags, based on co-occurrence
    statistics, since there is not a one-to-one mapping.
    """
    if pos[0] == "v" or pos in ["rp", "ex", "md"]:  # Verbs
        rule_pos = "v"
    elif pos[0] == "n" or pos in ["$", ":", "sym", "uh", "wp"]:  # Nouns
        rule_pos = "n"
    elif pos[0] == "j" or pos in ["ls", "pdt", "rbr", "rbs", "prp"]:  # Adjectives
        rule_pos = "a"
    elif pos == "cc":  # Conjunctions
        rule_pos = "c"
    elif pos in ["to", "in"]:  # Prepositions
        rule_pos = "prep"
    elif pos in ["dt", "wdt"]:  # Determinors
        rule_pos = "art"
    elif pos in ["rb", "wrb"]:  # Adverbs
        rule_pos = "adv"
    elif pos == "cd":  # Cardinal Numbers
        rule_pos = "num"
    else:
        sys.stderr.write("WARNING: Rule not defined for part-of-speech {} word {} - treating as noun.".format(pos, token))
        return "n"
    return rule_pos


def check_if_potential_target(lemma, target_lu_map):
    """
    Simple check to see if this is a potential position to even consider, based on
    the LU index provided under FrameNet. Note that since we use NLTK lemmas,
    this might be lossy.
    """
    nltk_lem_str = LEMDICT.getstr(lemma)
    return nltk_lem_str in target_lu_map or nltk_lem_str.lower() in target_lu_map


def create_lexical_unit(lemma_id, pos_id, token_id, target_lu_map, lu_names):
    """
    Given a lemma ID and a POS ID (both lemma and POS derived from NLTK),
    create a LexicalUnit object.
    If lemma is unknown, then check if token is in the LU vocabulary, and
    use it if present (Hack).
    """
    nltk_lem_str = LEMDICT.getstr(lemma_id)
    if nltk_lem_str not in target_lu_map and nltk_lem_str.lower() in target_lu_map:
        nltk_lem_str = nltk_lem_str.lower()

    # Lemma is not in FrameNet, but it could be a lemmatization error.
    if nltk_lem_str == UNK:
        if VOCDICT.getstr(token_id) in target_lu_map:
            nltk_lem_str = VOCDICT.getstr(token_id)
        elif VOCDICT.getstr(token_id).lower() in target_lu_map:
            nltk_lem_str = VOCDICT.getstr(token_id).lower()
    assert nltk_lem_str in target_lu_map
    assert LUDICT.getid(nltk_lem_str) != LUDICT.getid(UNK)

    nltk_pos_str = POSDICT.getstr(pos_id)
    rule_pos_str = get_fn_pos_by_rules(nltk_pos_str.lower(), nltk_lem_str)
    rule_lupos = nltk_lem_str + "." + rule_pos_str

    # Lemma is not seen with this pos tag.
    if rule_lupos not in lu_names:
        # Hack: replace with anything the lemma is seen with.
        rule_pos_str = list(target_lu_map[nltk_lem_str])[0].split(".")[-1]
    return LexicalUnit(LUDICT.getid(nltk_lem_str), LUPOSDICT.getid(rule_pos_str))


def identify_targets(builders, tokens, postags, lemmas, model_variables, gold_targets=None):
    """
    Target identification model, using bidirectional LSTMs, with a
    multilinear perceptron layer on top for classification.
    """

    v_x = model_variables['v_x']
    p_x = model_variables['p_x']
    l_x = model_variables['l_x']
    pretrained_map = model_variables['pretrained_map']
    e_x = model_variables['e_x']
    u_x = model_variables['u_x']
    w_e = model_variables['w_e']
    b_e = model_variables['b_e']
    USE_DROPOUT = model_variables['USE_DROPOUT']
    DROPOUT_RATE = model_variables['DROPOUT_RATE']
    target_lu_map = model_variables['target_lu_map']
    lu_names = model_variables['lu_names']
    w_f = model_variables['w_f']
    w_z = model_variables['w_z']
    b_z = model_variables['b_z']
    b_f = model_variables['b_f']

    renew_cg()
    train_mode = (gold_targets is not None)

    sentlen = len(tokens)
    emb_x = [v_x[tok] for tok in tokens]
    pos_x = [p_x[pos] for pos in postags]
    lem_x = [l_x[lem] for lem in lemmas]

    emb2_xi = []
    for i in xrange(sentlen):
        if tokens[i] in pretrained_map:
            # Prevent the pretrained embeddings from being updated.
            emb_without_backprop = lookup(e_x, tokens[i], update=False)
            features_at_i = concatenate([emb_x[i], pos_x[i], lem_x[i], emb_without_backprop])
        else:
            features_at_i = concatenate([emb_x[i], pos_x[i], lem_x[i], u_x])
        emb2_xi.append(w_e * features_at_i + b_e)

    emb2_x = [rectify(emb2_xi[i]) for i in xrange(sentlen)]

    # Initializing the two LSTMs.
    if USE_DROPOUT and train_mode:
        builders[0].set_dropout(DROPOUT_RATE)
        builders[1].set_dropout(DROPOUT_RATE)
    f_init, b_init = [i.initial_state() for i in builders]

    fw_x = f_init.transduce(emb2_x)
    bw_x = b_init.transduce(reversed(emb2_x))

    losses = []
    predicted_targets = {}
    for i in xrange(sentlen):
        if not check_if_potential_target(lemmas[i], target_lu_map):
            continue
        h_i = concatenate([fw_x[i], bw_x[sentlen - i - 1]])
        score_i = w_f * rectify(w_z * h_i + b_z) + b_f
        if train_mode and USE_DROPOUT:
            score_i = dropout(score_i, DROPOUT_RATE)

        logloss = log_softmax(score_i, [0, 1])
        if not train_mode:
            is_target = np.argmax(logloss.npvalue())
        else:
            is_target = int(i in gold_targets)

        if int(np.argmax(logloss.npvalue())) != 0:
            predicted_targets[i] = (create_lexical_unit(lemmas[i], postags[i], tokens[i], target_lu_map, lu_names), None)

        losses.append(pick(logloss, is_target))

    objective = -esum(losses) if losses else None
    return objective, predicted_targets


def print_as_conll_targetid(gold_examples, predicted_target_dict, out_conll_file):
    """
    Creates a CoNLL object with predicted target and lexical unit.
    Spits out one CoNLL for each LU.
    """
    
    with codecs.open(out_conll_file, "w", "utf-8") as conll_file:
        for gold, pred in zip(gold_examples, predicted_target_dict):
            for target in sorted(pred):
                result = gold.get_predicted_target_conll(target, pred[target][0]) + "\n"
                conll_file.write(result)
        conll_file.close()
    

def print_as_conll_frameid(goldexamples, pred_targmaps, out_conll_file):
    with codecs.open(out_conll_file, "w", "utf-8") as f:
        for g,p in zip(goldexamples, pred_targmaps):
            result = g.get_predicted_frame_conll(p) + "\n"
            f.write(result)
        f.close()










def build_targetid_model(model_name, common):
    

    model_dir = "logs/{}/".format(model_name)
    model_file_name = "{}best-targetid-{}-model".format(model_dir, VERSION)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_conll = TRAIN_FTE

    USE_DROPOUT = False

    sys.stderr.write("_____________________\n")
    sys.stderr.write("COMMAND: {}\n".format(" ".join(sys.argv)))
    sys.stderr.write("MODEL FOR TEST / PREDICTION:\t{}\n".format(model_file_name))
    sys.stderr.write("_____________________\n\n")

    PRETRAINED_DIM = common['PRETRAINED_DIM']
    pretrained_map = common['pretrained_map']
    target_lu_map = common['target_lu_map']
    lu_names = common['lu_names']


    # Default configurations.
    configuration = {"train": train_conll,
                    "unk_prob": 0.1,
                    "dropout_rate": 0.01,
                    "token_dim": 100,
                    "pos_dim": 100,
                    "lemma_dim": 100,
                    "lstm_input_dim": 100,
                    "lstm_dim": 100,
                    "lstm_depth": 2,
                    "hidden_dim": 100,
                    "use_dropout": USE_DROPOUT,
                    "pretrained_embedding_dim": PRETRAINED_DIM,
                    "num_epochs": 100,
                    "patience": 25,
                    "eval_after_every_epochs": 100,
                    "dev_eval_epoch_frequency": 3}
    configuration_file = os.path.join(model_dir, "configuration.json")
    json_file = open(configuration_file, "r")
    configuration = json.load(json_file)

    UNK_PROB = configuration["unk_prob"]
    DROPOUT_RATE = configuration["dropout_rate"]

    TOK_DIM = configuration["token_dim"]
    POS_DIM = configuration["pos_dim"]
    LEMMA_DIM = configuration["lemma_dim"]
    INPUT_DIM = TOK_DIM + POS_DIM + LEMMA_DIM

    LSTM_INP_DIM = configuration["lstm_input_dim"]
    LSTM_DIM = configuration["lstm_dim"]
    LSTM_DEPTH = configuration["lstm_depth"]
    HIDDEN_DIM = configuration["hidden_dim"]

    NUM_EPOCHS = configuration["num_epochs"]
    PATIENCE = configuration["patience"]
    EVAL_EVERY_EPOCH = configuration["eval_after_every_epochs"]
    DEV_EVAL_EPOCH = configuration["dev_eval_epoch_frequency"] * EVAL_EVERY_EPOCH

    sys.stderr.write("\nPARSER SETTINGS (see {})\n_____________________\n".format(configuration_file))
    for key in sorted(configuration):
        sys.stderr.write("{}:\t{}\n".format(key.upper(), configuration[key]))

    sys.stderr.write("\n")


    print_data_status(VOCDICT, "Tokens")
    print_data_status(POSDICT, "POS tags")
    print_data_status(LEMDICT, "Lemmas")
    sys.stderr.write("\n_____________________\n\n")







    model = Model()
    trainer = SimpleSGDTrainer(model, 0.01)

    v_x = model.add_lookup_parameters((VOCDICT.size(), TOK_DIM))
    p_x = model.add_lookup_parameters((POSDICT.size(), POS_DIM))
    l_x = model.add_lookup_parameters((LEMDICT.size(), LEMMA_DIM))

    e_x = model.add_lookup_parameters((VOCDICT.size(), PRETRAINED_DIM))
    for wordid in pretrained_map:
        e_x.init_row(wordid, pretrained_map[wordid])
    # Embedding for unknown pretrained embedding.
    u_x = model.add_lookup_parameters((1, PRETRAINED_DIM), init='glorot')

    w_e = model.add_parameters((LSTM_INP_DIM, PRETRAINED_DIM + INPUT_DIM))
    b_e = model.add_parameters((LSTM_INP_DIM, 1))

    w_i = model.add_parameters((LSTM_INP_DIM, INPUT_DIM))
    b_i = model.add_parameters((LSTM_INP_DIM, 1))

    builders = [
        LSTMBuilder(LSTM_DEPTH, LSTM_INP_DIM, LSTM_DIM, model),
        LSTMBuilder(LSTM_DEPTH, LSTM_INP_DIM, LSTM_DIM, model),
    ]

    w_z = model.add_parameters((HIDDEN_DIM, 2*LSTM_DIM))
    b_z = model.add_parameters((HIDDEN_DIM, 1))
    w_f = model.add_parameters((2, HIDDEN_DIM))  # prediction: is a target or not.
    b_f = model.add_parameters((2, 1))




    model_variables = {
        'v_x': v_x,
        'p_x': p_x,
        'l_x': l_x,
        'pretrained_map': pretrained_map,
        'e_x': e_x,
        'u_x': u_x,
        'w_e': w_e,
        'b_e': b_e,
        'USE_DROPOUT': USE_DROPOUT,
        'DROPOUT_RATE': DROPOUT_RATE,
        'target_lu_map': target_lu_map,
        'lu_names': lu_names,
        'w_f': w_f,
        'w_z': w_z,
        'b_z': b_z,
        'b_f': b_f,
        'builders': builders,
        'model_dir': model_dir
    }


    sys.stderr.write("Reading model from {} ...\n".format(model_file_name))
    model.populate(model_file_name)

    return model, model_variables

def run_model_targetid(raw_input_path, model_variables):

    builders = model_variables['builders']
    model_dir = model_variables['model_dir']

    with open(raw_input_path, "r") as fin:
        instances = []
        offsets = [] # contains array with for each row, the offsets of each token
        for i,line in enumerate(fin):
            instance, offset = make_data_instance(line, i, get_offsets=True)
            instances.append(instance)
            offsets.append(offset)
        # filter empty rows
        instances = [el for el in instances if el]

    with open('{}offsets.json'.format(model_dir), 'w') as f:
        json.dump(offsets, f)
    out_conll_file = "{}predicted-targets.conll".format(model_dir)

    predictions = []
    for instance in instances:
        _, prediction = identify_targets(builders, instance.tokens, instance.postags, instance.lemmas, model_variables)
        predictions.append(prediction)
    sys.stderr.write("Printing output in CoNLL format to {}\n".format(out_conll_file))
    print_as_conll_targetid(instances, predictions, out_conll_file)
    sys.stderr.write("Done!\n")

    return offsets


def find_multitokentargets(examples, split):
    multitoktargs = tottargs = 0.0
    for tr in examples:
        tottargs += 1
        if len(tr.targetframedict) > 1:
            multitoktargs += 1
            tfs = set(tr.targetframedict.values())
            if len(tfs) > 1:
                raise Exception("different frames for neighboring targets!", tr.targetframedict)
    sys.stderr.write("multi-token targets in %s: %.3f%% [%d / %d]\n"
                     %(split, multitoktargs*100/tottargs, multitoktargs, tottargs))


def identify_frames(builders, tokens, postags, lexunit, targetpositions, model_variables, goldframe=None):
    
    v_x = model_variables['v_x']
    p_x = model_variables['p_x']
    pretrained_embeddings_map = model_variables['pretrained_embeddings_map']
    e_x = model_variables['e_x']
    u_x = model_variables['u_x']
    w_e = model_variables['w_e']
    b_e = model_variables['b_e']
    USE_DROPOUT = model_variables['USE_DROPOUT']
    DROPOUT_RATE = model_variables['DROPOUT_RATE']
    tlstm = model_variables['tlstm']
    lufrmmap = model_variables['lufrmmap']
    USE_HIER = model_variables['USE_HIER']
    relatedlus = model_variables['relatedlus']
    lu_x = model_variables['lu_x']
    lp_x = model_variables['lp_x']
    w_f = model_variables['w_f']
    w_z = model_variables['w_z']
    b_z = model_variables['b_z']
    b_f = model_variables['b_f']
    
    renew_cg()
    trainmode = (goldframe is not None)

    sentlen = len(tokens) - 1
    emb_x = [v_x[tok] for tok in tokens]
    pos_x = [p_x[pos] for pos in postags]

    emb2_xi = []
    for i in xrange(sentlen + 1):
        if tokens[i] in pretrained_embeddings_map:
            # If update set to False, prevents pretrained embeddings from being updated.
            emb_without_backprop = lookup(e_x, tokens[i], update=True)
            features_at_i = concatenate([emb_x[i], pos_x[i], emb_without_backprop])
        else:
            features_at_i = concatenate([emb_x[i], pos_x[i], u_x])
        emb2_xi.append(w_e * features_at_i + b_e)

    emb2_x = [rectify(emb2_xi[i]) for i in xrange(sentlen+1)]

    # initializing the two LSTMs
    if USE_DROPOUT and trainmode:
        builders[0].set_dropout(DROPOUT_RATE)
        builders[1].set_dropout(DROPOUT_RATE)
    f_init, b_init = [i.initial_state() for i in builders]

    fw_x = f_init.transduce(emb2_x)
    bw_x = b_init.transduce(reversed(emb2_x))

    # only using the first target position - summing them hurts :(
    targetembs = [concatenate([fw_x[targetidx], bw_x[sentlen - targetidx - 1]]) for targetidx in targetpositions]
    targinit = tlstm.initial_state()
    target_vec = targinit.transduce(targetembs)[-1]

    valid_frames = list(lufrmmap[lexunit.id])
    chosenframe = valid_frames[0]
    logloss = None
    if len(valid_frames) > 1:
        if USE_HIER and lexunit.id in relatedlus:
            lu_vec = esum([lu_x[luid] for luid in relatedlus[lexunit.id]])
        else:
            lu_vec = lu_x[lexunit.id]
        fbemb_i = concatenate([target_vec, lu_vec, lp_x[lexunit.posid]])
        # TODO(swabha): Add more Baidu-style features here.
        f_i = w_f * rectify(w_z * fbemb_i + b_z) + b_f
        if trainmode and USE_DROPOUT:
            f_i = dropout(f_i, DROPOUT_RATE)

        logloss = log_softmax(f_i, valid_frames)

        if not trainmode:
            chosenframe = np.argmax(logloss.npvalue())

    if trainmode:
        chosenframe = goldframe.id

    losses = []
    if logloss is not None:
        losses.append(pick(logloss, chosenframe))

    prediction = {tidx: (lexunit, Frame(chosenframe)) for tidx in targetpositions}

    objective = -esum(losses) if losses else None
    return objective, prediction


def build_frameid_model(options, common):
    model_dir = "logs/{}/".format(options['model_name'])
    model_file_name = "{}best-frameid-{}-model".format(model_dir, VERSION)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_conll = TRAIN_FTE
    USE_DROPOUT = False
    USE_WV = True
    USE_HIER = options.get('hier', False)

    sys.stderr.write("_____________________\n")
    sys.stderr.write("COMMAND: {}\n".format(" ".join(sys.argv)))
    sys.stderr.write("MODEL FOR TEST / PREDICTION:\t{}\n".format(model_file_name))
    sys.stderr.write("_____________________\n\n")

    lufrmmap = common['lufrmmap']
    relatedlus = common['relatedlus']
    if USE_WV:
        pretrained_embeddings_map = common['pretrained_map']
        PRETRAINED_DIM = len(pretrained_embeddings_map.values()[0])

    # Default configurations.
    configuration = {'train': train_conll,
                    'use_exemplar': options['exemplar'],
                    'use_hierarchy': USE_HIER,
                    'unk_prob': 0.1,
                    'dropout_rate': 0.01,
                    'token_dim': 100,
                    'pos_dim': 100,
                    'lu_dim': 100,
                    'lu_pos_dim': 100,
                    'lstm_input_dim': 100,
                    'lstm_dim': 100,
                    'lstm_depth': 2,
                    'hidden_dim': 100,
                    'use_dropout': USE_DROPOUT,
                    'pretrained_embedding_dim': PRETRAINED_DIM,
                    'num_epochs': 100 if not options['exemplar'] else 25,
                    'patience': 25,
                    'eval_after_every_epochs': 100,
                    'dev_eval_epoch_frequency': 5}
    configuration_file = os.path.join(model_dir, 'configuration.json')

    json_file = open(configuration_file, "r")
    configuration = json.load(json_file)


    UNK_PROB = configuration['unk_prob']
    DROPOUT_RATE = configuration['dropout_rate']

    TOKDIM = configuration['token_dim']
    POSDIM = configuration['pos_dim']
    LUDIM = configuration['lu_dim']
    LPDIM = configuration['lu_pos_dim']
    INPDIM = TOKDIM + POSDIM

    LSTMINPDIM = configuration['lstm_input_dim']
    LSTMDIM = configuration['lstm_dim']
    LSTMDEPTH = configuration['lstm_depth']
    HIDDENDIM = configuration['hidden_dim']

    NUM_EPOCHS = configuration['num_epochs']
    PATIENCE = configuration['patience']
    EVAL_EVERY_EPOCH = configuration['eval_after_every_epochs']
    DEV_EVAL_EPOCH = configuration['dev_eval_epoch_frequency'] * EVAL_EVERY_EPOCH

    sys.stderr.write("\nPARSER SETTINGS (see {})\n_____________________\n".format(configuration_file))
    for key in sorted(configuration):
        sys.stderr.write("{}:\t{}\n".format(key.upper(), configuration[key]))

    sys.stderr.write("\n")

    print_data_status(VOCDICT, "Tokens")
    print_data_status(POSDICT, "POS tags")
    print_data_status(LUDICT, "LUs")
    print_data_status(LUPOSDICT, "LU POS tags")
    print_data_status(FRAMEDICT, "Frames")
    sys.stderr.write("\n_____________________\n\n")

    model = Model()
    trainer = SimpleSGDTrainer(model)
    # trainer = AdamTrainer(model, 0.0001, 0.01, 0.9999, 1e-8)

    v_x = model.add_lookup_parameters((VOCDICT.size(), TOKDIM))
    p_x = model.add_lookup_parameters((POSDICT.size(), POSDIM))
    lu_x = model.add_lookup_parameters((LUDICT.size(), LUDIM))
    lp_x = model.add_lookup_parameters((LUPOSDICT.size(), LPDIM))
    if USE_WV:
        e_x = model.add_lookup_parameters((VOCDICT.size(), PRETRAINED_DIM))
        for wordid in pretrained_embeddings_map:
            e_x.init_row(wordid, pretrained_embeddings_map[wordid])

        # Embedding for unknown pretrained embedding.
        u_x = model.add_lookup_parameters((1, PRETRAINED_DIM), init='glorot')

        w_e = model.add_parameters((LSTMINPDIM, PRETRAINED_DIM+INPDIM))
        b_e = model.add_parameters((LSTMINPDIM, 1))

    w_i = model.add_parameters((LSTMINPDIM, INPDIM))
    b_i = model.add_parameters((LSTMINPDIM, 1))

    builders = [
        LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model),
        LSTMBuilder(LSTMDEPTH, LSTMINPDIM, LSTMDIM, model),
    ]

    tlstm = LSTMBuilder(LSTMDEPTH, 2*LSTMDIM, LSTMDIM, model)

    w_z = model.add_parameters((HIDDENDIM, LSTMDIM + LUDIM + LPDIM))
    b_z = model.add_parameters((HIDDENDIM, 1))
    w_f = model.add_parameters((FRAMEDICT.size(), HIDDENDIM))
    b_f = model.add_parameters((FRAMEDICT.size(), 1))


    model_variables = {
        'v_x': v_x,
        'p_x': p_x,
        'pretrained_embeddings_map': pretrained_embeddings_map,
        'e_x': e_x,
        'u_x': u_x,
        'w_e': w_e,
        'b_e': b_e,
        'USE_DROPOUT': USE_DROPOUT,
        'DROPOUT_RATE': DROPOUT_RATE,
        'tlstm': tlstm,
        'lufrmmap': lufrmmap,
        'USE_HIER': USE_HIER,
        'relatedlus': relatedlus,
        'lu_x': lu_x,
        'lp_x': lp_x,
        'w_f': w_f,
        'w_z': w_z,
        'b_z': b_z,
        'b_f': b_f,
        'builders': builders,
        'model_dir': model_dir
    }


    sys.stderr.write("Loading model from {} ...\n".format(model_file_name))
    model.populate(model_file_name)

    return model, model_variables


def run_model_frameid(options, model_variables):

    assert options['raw_input'] is not None
    instances, _, _ = read_conll(options['raw_input'])

    builders = model_variables['builders']
    model_dir = model_variables['model_dir']

    out_conll_file = "{}predicted-frames.conll".format(model_dir)

    predictions = []
    sents_num = []
    for instance in instances:
        _, prediction = identify_frames(builders, instance.tokens, instance.postags, instance.lu, instance.targetframedict.keys(), model_variables)
        sents_num.append(instance.sent_num)
        predictions.append(prediction)
    sys.stderr.write("Printing output in CoNLL format to {}\n".format(out_conll_file))
    print_as_conll_frameid(instances, predictions, out_conll_file)
    sys.stderr.write("Done!\n")
    return predictions, sents_num


if __name__ == '__main__':

    optpr = OptionParser()
    optpr.add_option("-n", "--model_name", help="Name of model directory to save model to.")
    optpr.add_option("--raw_input", type="str", metavar="FILE")
    (options, args) = optpr.parse_args()

    common = common_load()

    model_name_targetid = 'fn1.7-pretrained-targetid'
    raw_input = options.raw_input
    options_frameid = {
        'model_name': 'fn1.7-pretrained-frameid',
        'raw_input': 'logs/fn1.7-pretrained-targetid/predicted-targets.conll',
        'hier': False,
        'exemplar': False
    }

    model_targetid, model_targetid_variables = build_targetid_model(model_name_targetid, common)
    model_frameid, model_frameid_variables = build_frameid_model(options_frameid, common)
    
    offsets = run_model_targetid(raw_input, model_targetid_variables)
    predictions, sents_num = run_model_frameid(options_frameid, model_frameid_variables)

    # [{index: (LexicalUnit, Frame)} for each instance]
    # LexicalUnit.get_str(LUDICT, LUPOSDICT), Frame.get_str(FRAMEDICT)
    #print(predictions)
    result = []
    for prediction, sent_num in zip(predictions, sents_num):
        for token_id, (lu, frame) in prediction.items():
            offset_line = offsets[sent_num]
            lu = {
                'lu': lu.get_str(LUDICT, LUPOSDICT),
                'frame': frame.get_str(FRAMEDICT),
                'offset': {
                    'line': sent_num,
                    'start': offset_line[token_id][0],
                    'end': offset_line[token_id][1]
                }
            }
            result.append(lu)

    with open('predictions.json', 'w') as f:
        json.dump(result, f)
