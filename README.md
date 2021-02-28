# sig2vec (A.K.A GOD)

Initial setup for **G**lobal **O**bject Embe**d**der.

# Setup

After cloning, navigate into the working directory and call

```bash
mkdir data models plots raw_data
pip install -r requirements.txt
```

After this, move the data embedding data you want to re-format for the Variational Autoencoder (VAE) into `raw_data` and call

```bash
python format_data
```

This will reformat the data and save it as a pickle to `data`. After this is done, simply call

```bash
python vae.py
```

To use the VAE on the re-formatted data. The model will save its weights in `models` after training. This code will also use `HDBSCAN` to detect groups in the data, and plot a 2D projection of a testing data sub-sample to `plots`.

## Working example

If you want something that will actually work, use `test_vae.py` which trains a VAE on the MNIST dataset.