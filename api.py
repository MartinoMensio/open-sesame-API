import marshmallow
import os
from flask import Flask
from flask import request
from flask_restx import Resource, Api
# from webargs.flaskparser import use_args, use_kwargs

from sesame import pipeline

p = pipeline.get_pipeline()

app = Flask(__name__)
api = Api(app)

@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}
        

@api.route('/parse')
class Parse(Resource):
    args = {
        'text': marshmallow.fields.Str(),
    }

    # @api.expect('sentences', 'The sentences to analyse')
    # @use_kwargs(args)
    # @api.expect(api.model)
    def post(self):
        #print(sentences)
        # get request sentences
        body = request.json
        print(body.keys())
        text = body['text']

        # run parsing

        result = pipeline.run_pipeline(p, text)
        return result


