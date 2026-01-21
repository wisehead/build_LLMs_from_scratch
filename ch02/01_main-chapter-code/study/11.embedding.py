import torch
input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

#print the weight matrix
print("Weight matrix:")
print(embedding_layer.weight)

#print the embedding of the token with id 3
print("Embedding of token with id 3:")
print(embedding_layer(torch.tensor([3])))

#print the embeddings of the input ids
print("Embeddings of input ids:")
print(embedding_layer(input_ids))