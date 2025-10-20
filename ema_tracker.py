import torch

class EMA:
    def __init__(self, beta=0.8):
        self.beta = beta
        self.vals = {}
        self.cross_scores = {}

    def update_layer(self, client_id, key, x):
        k = (client_id, key)
        if k not in self.vals:
            self.vals[k] = x.detach().clone()
        else:
            self.vals[k] = self.beta*self.vals[k] + (1-self.beta)*x
        return self.vals[k]

    def update_cross(self, client_id, score):
        if client_id not in self.cross_scores:
            self.cross_scores[client_id] = score
        else:
            self.cross_scores[client_id] = self.beta*self.cross_scores[client_id] + (1-self.beta)*score
        return self.cross_scores[client_id]
