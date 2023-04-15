import torch
import pickle


def load_predictor_ccl(model, model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q') and 'fc' not in k:
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        elif k.startswith('module.encoder_q') and 'fc' not in k:
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    return model


def get_arch_info(arch_path):
    # arch_path = "1e4d24eb52f67424eabfe070ffbaee7ac2f31ca4f2e19a3c87680fbb4ed8167a/
    # 1e4d24eb52f67424eabfe070ffbaee7ac2f31ca4f2e19a3c87680fbb4ed8167a.pkl"
    with open(arch_path, "rb") as f:
        arch = pickle.load(f)
    return arch