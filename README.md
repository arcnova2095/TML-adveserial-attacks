The main idea of the adversarial training is pretty straightforward: generate the adversarial examples
during the training and include them in each batch so that the model learns to be correct under attack
too. Concretely, for every training batch we:
1. Run PGD on the batch using the current model to produce the adversarial examples.
2. Concatenate the original clean images and the adversarial images (doubling the batch size).
3. Do a forward pass on this combined batch, compute cross-entropy, and update the weights.
The fact that adversarial examples are generated fresh every epoch using the current model is im-
portant, the model is always being challenged by its own current blind spots rather than a fixed set
of precomputed perturbation
