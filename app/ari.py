import itertools
from datasets import Dataset
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json

TOKENIZER = AutoTokenizer.from_pretrained("raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L")
MODEL = AutoModelForSequenceClassification.from_pretrained("raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L")


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


def make_predictions(trainer, tknz_data):
    predicted_logprobs = trainer.predict(tknz_data)
    predicted_labels = np.argmax(predicted_logprobs.predictions, axis=-1)

    return predicted_labels


def output_xaif(idents, labels, fileaif):
    newnodeId = 90000
    newedgeId = 80000
    for i in range(len(labels)):
        lb = labels[i]

        if lb == 0:
            continue

        elif lb == 1:
            # Add the RA node
            fileaif["AIF"]["nodes"].append({"nodeID": newnodeId, "text": "Default Inference", "type": "RA", "timestamp": "", "scheme": "Default Inference", "schemeID": "72"})

            # Add the edges from ident[0] to RA and from RA to ident[1]
            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["AIF"]["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["AIF"]["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 2:
            # Add the CA node
            fileaif["AIF"]["nodes"].append({"nodeID": newnodeId, "text": "Default Conflict", "type": "CA", "timestamp": "", "scheme": "Default Conflict", 'schemeID': "71"})

            # Add the edges from ident[0] to MA and from MA to ident[1]
            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["AIF"]["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["AIF"]["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 3:
            # Add the MA node
            fileaif["AIF"]["nodes"].append({"nodeID": newnodeId, "text": "Default Rephrase", "type": "MA", "timestamp": "", 'scheme': "Default Rephrase", 'schemeID': "144"})

            # Add the edges from ident[0] to MA and from MA to ident[1]
            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["AIF"]["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["AIF"]["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

    return fileaif


def relation_identification(xaif, window_size):

    # Generate a HF Dataset from all the "I" node pairs to make predictions from the xAIF file 
    # and a list of tuples with the corresponding "I" node ids to generate the final xaif file.
    dataset, ids, props = preprocess_data(xaif['AIF'], window_size)

    # Tokenize the Dataset.
    tokenized_data = dataset.map(tokenize_sequence, batched=True)

    # Instantiate HF Trainer for predicting.
    trainer = Trainer(MODEL)

    # Predict the list of labels for all the pairs of "I" nodes.
    labels = make_predictions(trainer, tokenized_data)

    # Prepare the xAIF output file.
    out_xaif = output_xaif(ids, labels, xaif)

    return out_xaif


# DEBUGGING:
if __name__ == "__main__":
    ff = open('../data.json', 'r')
    content = json.load(ff)
    print(content)
    out = relation_identification(content, 3)
    with open("../data_out.json", "w") as outfile:
        json.dump(out, outfile, indent=4)

