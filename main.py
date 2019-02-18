#!/usr/bin/env python3

import win_unicode_console
win_unicode_console.enable()

import json
import numpy as np
import tensorflow as tf

import sys, os
sys.path.append(os.path.join(sys.path[0], 'src'))
import model, sample, encoder

from flask import Flask, request, jsonify


model_name = '117M'
seed = None
nsamples = 1
batch_size = None
length = None
temperature = 1
top_k = 0

if batch_size is None:
    batch_size = 1
assert nsamples % batch_size == 0
np.random.seed(seed)
tf.set_random_seed(seed)

enc = encoder.get_encoder(model_name)
hparams = model.default_hparams()
with open(os.path.join('models', model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

length = 64

g = tf.Graph()
with g.as_default():
    context = tf.placeholder(tf.int32, [batch_size, None])
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k
    )
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
sess = tf.Session(graph=g)
saver.restore(sess, ckpt)


app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('./index.html')

@app.route('/gen', methods=['POST'])
def gen():
    content = request.get_json()
    seed = content.get('seed')
    raw_text = seed
    if not raw_text:
        raw_text = ' '
    print('Model prompt >>> ' + raw_text)
    context_tokens = enc.encode(raw_text)
    out = sess.run(output, feed_dict={
        context: [context_tokens]
    })[:, len(context_tokens):]
    text = enc.decode(out[0])
    print('Generated >>> ' + text)
    return jsonify({'text': text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

