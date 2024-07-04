import itertools
from datasets import Dataset
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import json
import torch
from torch.nn import functional as F
from amf_fast_inference import model
from xaif_eval import xaif


TOKENIZER = AutoTokenizer.from_pretrained("raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L")
# MODEL = AutoModelForSequenceClassification.from_pretrained("raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L")
MODEL_ID = "raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L"
LOADER = model.ModelLoader(MODEL_ID)
PRUNED_MODEL = LOADER.load_model()


def preprocess_data(filexaif, wnd_size):
    idents = []
    idents_comb = []
    propositions = {}
    data = {'text': [], 'text2': []}

    for node in filexaif['nodes']:
        if node['type'] == 'I':
            propositions[node['nodeID']] = node['text']
            idents.append(node['nodeID'])

    if wnd_size == -1:
        window_size = len(idents)
    else:
        window_size = wnd_size
    for i in range(len(idents) - window_size + 1):
        context = idents[i: i + window_size]

        if window_size == 2:
            for p in itertools.combinations(context, 2):
                idents_comb.append(p)
                data['text'].append(propositions[p[0]])
                data['text2'].append(propositions[p[1]])

        else:
            if i == 0:
                c = 1
            else:
                c = 0
            for p in itertools.combinations(context, 2):
                if c == 0:
                    pass
                else:
                    idents_comb.append(p)
                    data['text'].append(propositions[p[0]])
                    data['text2'].append(propositions[p[1]])
                c += 1

    final_data = Dataset.from_dict(data)

    return final_data, idents_comb, propositions


def tokenize_sequence(samples):
    return TOKENIZER(samples["text"], samples["text2"], padding="max_length", truncation=True)


def pipeline_predictions(pipeline, data):
    labels = []
    pipeline_input = []
    for i in range(len(data['text'])):
        sample = data['text'][i]+'. '+data['text2'][i]
        pipeline_input.append(sample)

    outputs = pipeline(pipeline_input)
    for out in outputs:
        if out['label'] == 'Inference' and out['score'] > 0.95:
            labels.append(1)
        elif out['label'] == 'Conflict' and out['score'] > 0.8:
            labels.append(2)
        elif out['label'] == 'Rephrase' and out['score'] > 0.8:
            labels.append(3)
        else:
            labels.append(0)

    return labels


def make_predictions(trainer, tknz_data):
    predicted_logprobs = trainer.predict(tknz_data)
    '''
        predicted_labels = np.argmax(predicted_logprobs.predictions, axis=-1)

        return predicted_labels
    '''
    labels = []
    for sample in predicted_logprobs.predictions:
        torch_logits = torch.from_numpy(sample)
        probabilities = F.softmax(torch_logits, dim=-1).numpy()
        valid_check = probabilities > 0.95
        if True in valid_check:
            labels.append(np.argmax(sample, axis=-1))
        elif np.argmax(sample, axis=-1) == 2 and probabilities[2] > 0.8:
            labels.append(2)
        else:
            labels.append(-1)

    return labels


def output_xaif(idents, labels, fileaif):
    original_aif = xaif.AIF(fileaif)

    for i in range(len(labels)):
        lb = labels[i]

        if lb == 0:
            continue

        elif lb == 1:
            # Add the RA node
            original_aif.add_component("argument_relation", "RA", idents[i][1], idents[i][0])

        elif lb == 2:
            # Add the CA node
            original_aif.add_component("argument_relation", "CA", idents[i][1], idents[i][0])

        elif lb == 3:
            # Add the MA node
            original_aif.add_component("argument_relation", "MA", idents[i][1], idents[i][0])

    return original_aif.xaif


def relation_identification(xaif, window_size):

    # Generate a HF Dataset from all the "I" node pairs to make predictions from the xAIF file 
    # and a list of tuples with the corresponding "I" node ids to generate the final xaif file.
    dataset, ids, props = preprocess_data(xaif['AIF'], window_size)

    # Inference Pipeline
    pl = pipeline("text-classification", model=PRUNED_MODEL, tokenizer=TOKENIZER)

    # Predict the list of labels for all the pairs of "I" nodes.
    labels = pipeline_predictions(pl, dataset)

    # Prepare the xAIF output file.
    out_xaif = output_xaif(ids, labels, xaif)

    return out_xaif


# DEBUGGING:
if __name__ == "__main__":
    ff = open('../data.json', 'r')
    content = json.load(ff)
    # print(content)
    out = relation_identification(content, -1)
    with open("../data_out3.json", "w") as outfile:
        json.dump(out, outfile, indent=4)

