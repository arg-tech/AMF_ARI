from flask import redirect, request, render_template, jsonify
from . import application
import json
from app.ari import relation_identification


@application.route('/amf_ari', methods=['GET', 'POST'])
def amf_schemes():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        ff = open(f.filename, 'r')
        content = json.load(ff)
        # Predict existing relations in content (i.e., xaif file) "I" nodes.
        response = relation_identification(content)
        print(response)
        return jsonify(response)
        
 
 
