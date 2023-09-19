from flask import redirect, request, render_template, jsonify
from . import application
import json
from app.ari import relation_identification


@application.route('/', methods=['GET', 'POST'])
def amf_schemes():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        ff = open(f.filename, 'r')
        content = json.load(ff)
        # Predict existing relations in content (i.e., xaif file) "I" nodes.
        window_size = request.args.get('window_size')
        if window_size is None:
            response = relation_identification(content, window_size=-1)
        else:
            response = relation_identification(content, window_size=window_size)
        print(response)
        return jsonify(response)
    elif request.method == 'GET':
        return render_template('docs.html')
 
 
