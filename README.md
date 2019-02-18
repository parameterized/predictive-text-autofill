
# predictive-text-autofill

Generate text template completions with a language model

Uses OpenAI's [gpt-2 code](https://github.com/openai/gpt-2)

## Installation

1. Download the model data

Follow instructions [here](https://cloud.google.com/storage/docs/gsutil_install) to install gsutil
```
sh download_model.sh 117M
```
If on Windows, run these commands in the repo directory
```
md models\117M
gsutil cp gs://gpt-2/models/117M/checkpoint models/117M
gsutil cp gs://gpt-2/models/117M/encoder.json models/117M
gsutil cp gs://gpt-2/models/117M/hparams.json models/117M
gsutil cp gs://gpt-2/models/117M/model.ckpt.data-00000-of-00001 models/117M
gsutil cp gs://gpt-2/models/117M/model.ckpt.index models/117M
gsutil cp gs://gpt-2/models/117M/model.ckpt.meta models/117M
gsutil cp gs://gpt-2/models/117M/vocab.bpe models/117M
```

2. Install python packages:
```
pip3 install -r requirements.txt
```
or
```
python -m pip install -r requirements.txt
```
(tested with python 3)

## Running

```
python main.py
```
Starts a local server on port 8080

Open a browser window and go to `localhost:8080`