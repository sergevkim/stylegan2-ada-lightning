import torch
import numpy as np


def calc_direction_split(model, mimic_layer):
    vectors = []
    for i in range(max(mimic_layer)):
        w1 = model.synthesis.blocks[i].conv0.affine.weight.data.cpu().numpy()
        w2 = model.synthesis.blocks[i].conv1.affine.weight.data.cpu().numpy()
        w = np.concatenate((w1,w2), axis=0).T
        w /= np.linalg.norm(w, axis=0, keepdims=True)
        _, eigen_vectors = np.linalg.eig(w.dot(w.T))
        vectors.append(torch.from_numpy(eigen_vectors[:,:5].T))
    return torch.cat(vectors, dim=0)   # (5*L) * 512


def compute_offsets(vectors, offset_weight, batch, device):
    num_direction = vectors.size(0)
    index = np.random.choice(np.arange(num_direction), size=(batch,)).astype(np.int64)
    offsets = vectors[index].to(device)

    norm = torch.norm(offsets, dim=1, keepdim=True)
    weight = torch.randn(batch, 1, device=device) * offset_weight

    offsets = offsets / norm * weight
    offsets = offsets[:, None, :]

    return offsets
