
import os
import os.path as osp

#Loading VPoser Body Pose Prior
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

def get_vposer_model(device='cuda', vposer_ckpt=None):
    # vposer_ckpt = osp.expandvars(vposer_ckpt)
    # you should get `vposer_ckpt` from https://huggingface.co/lithiumice/vposer
    # from huggingface_hub import snapshot_download

    vp, _ = load_model(vposer_ckpt, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
    vp = vp.to(device=device)
    vp.eval()
    return vp

if __name__ == '__main__':
    get_vposer_model()


