# Project for Introduction to Deep Learning
This repo contains my implementation and data for my project for my *Introduction to Deep Learning class*.

The goal of this project was to 1. Use Autoencoders to map a dataset to a latent space, 2. Use K-means on the latent space and then 3. compare the true labels and the clustered labels to see if the pipeline managed to retrieve the true labels.
The idea came from the paper Unsupervised Deep Embedding for Clustering Analysis https://arxiv.org/pdf/1511.06335, which made me want to explore further AE architectures to analyse the results and perhaps achieve better scores.

The project was documented in a report that can be found in the repo root.

MNIST was the only dataset that was utilized in this project, for simplicity reasons and because the mentioned paper also used it allowing me to draw direct comparisons in my analysis.

I implemented increasingly better performing Stacked Autoencoders (SAE), Autoencoders using Convolutional Layers and Variational Autoencoder (VAE).
The different architectures are separated in different notebooks. All show the architecture, the training, the final embedding, the k-means application and at the end various metrics to assess how well the resulting clusters resemble the true labels.
Many samples of the reconstruction of the AE's are shown in the notebook.
