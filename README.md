# Midterm - Plant Village Classification (RGB vs Grayscale)

## Instalação
pip install -r requirements.txt

## Treino
# MLP RGB
python midterm-nn-/src/train.py --model mlp --epochs 50 --seed 42 --color_mode rgb

# CNN RGB
python midterm-nn-/src/train.py --model cnn --epochs 50 --seed 42 --color_mode rgb

# CNN Grayscale
python midterm-nn-/src/train.py --model cnn --epochs 10 --seed 42 --color_mode grayscale

## App interativo
streamlit run src/app.py -- --model-path checkpoints/cnn_best.h5 --color-mode rgb
