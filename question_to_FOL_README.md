


# SETUP

## pre-reqs
pip install amrlib penman amr-logic-converter torch
wget https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz
tar -xzf model_parse_xfm_bart_large-v0_1_0.tar.gz
mkdir -p ~/amrlib_models
ln -s $(pwd)/model_parse_xfm_bart_large-v0_1_0 ~/amrlib_models/model_stog

## test your AMRLIB download was successful
python - <<EOF
import amrlib
stog = amrlib.load_stog_model(model_dir="~/amrlib_models/model_stog")
print(stog.parse_sents(["The cat sat on the mat."]))
EOF

## run the script
python question_to_FOL.py   hotpotqa_json/validation.json     --model-dir ~/amrlib_models/model_stog

## NOTE: the script is WIP; so far, it only prints the raw AMR Lib representation of each input sentence. Extra steps are needed to convert this to full FOL format.

