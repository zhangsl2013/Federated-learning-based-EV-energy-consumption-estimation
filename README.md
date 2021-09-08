# Federated-learning-based-EV-energy-consumption-estimation

The code in this repository provides energy consumption estimation on the data from, using the structure of federated learning.
During the training of the estimation model, calculated gradients from local workers (or individual vehicles in the network) are restricted to a unit norm that, the contribution of each data to the model is limited, so as to limit the risk of vehicle information disclosure from the trained model.
