import torch
from flat_model import ModelHolder
from config import get_config
from dataloader import d2v_pairer

class TrainedModel():
    def __init__(self, model_file_path, config_file_path):
        state_dict = torch.load(model_file_path)
        self.model = ModelHolder(cfg_all=get_config(cfg_file=config_file_path))
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.embed_meta = None
        self.pos_enc = None

    def fit(self, xs_meta, ys_meta):
        pass

    def predict(self, xs_target):
        if self.embed_meta is None or self.pos_enc is None:
            raise Exception("Model has not been fitted yet!")
        
        xs_target = xs_target.unsqueeze(0)
        with torch.no_grad():
            ys_pred_target = self.model.forward_target(xs_target, self.embed_meta, self.pos_enc)

        return torch.argmax(ys_pred_target.view(-1, 2), dim=1).numpy()

class FLAT(TrainedModel):
    def __init__(self, model_file_path, config_file_path):
        super().__init__(model_file_path, config_file_path)

    def fit(self, xs_meta, ys_meta):
        xs_meta, ys_meta = xs_meta.unsqueeze(0), ys_meta.unsqueeze(0)

        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        with torch.no_grad():
            self.embed_meta, self.pos_enc = self.model.forward_meta(pairs_meta)

    def __repr__(self):
        return "FLAT"
    
class FLATadapt(TrainedModel):
    def __init__(self, model_file_path, config_file_path):
        super().__init__(model_file_path, config_file_path)

    def fit(self, xs_meta, ys_meta):
        xs_meta, ys_meta = xs_meta.unsqueeze(0), ys_meta.unsqueeze(0)
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        with torch.no_grad():
            embed_meta, pos_enc = self.model.forward_meta(pairs_meta)

        embed_meta.requires_grad = True
        pos_enc.requires_grad = True
        optim_pos = torch.optim.Adam([pos_enc], lr=0.001)
        optim_embed = torch.optim.Adam([embed_meta], lr=0.075)
        for _ in range(5):
            # Make predictions on meta set and calc loss
            preds = self.model.forward_target(xs_meta, embed_meta, pos_enc)
            loss = torch.nn.functional.cross_entropy(preds.squeeze(), ys_meta.long().squeeze())
            loss.backward()
            optim_pos.step()
            optim_embed.step()
            optim_embed.zero_grad()
            optim_pos.zero_grad()

        self.embed_meta = embed_meta
        self.pos_enc = pos_enc

    def __repr__(self):
        return "FLATadapt"