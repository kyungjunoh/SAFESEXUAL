"""
Quality evaluation for generated images: FID (vs a reference set) + CLIP score
(prompt-image alignment). Self-contained: uses torchvision InceptionV3 for FID
features and open_clip for CLIP score (no torchmetrics / pytorch-fid needed).

Images are expected to be named "{case_number}_*.png" (as saved by generate_safree.py
into <save-dir>/all/). The prompt for each image is looked up from the COCO csv by
case_number, so CLIP score uses the true prompt.

Example:
  python eval_quality.py \
    --gen_dir results/gen_COCO_pcdd_rank40/all \
    --ref_dir results/gen_COCO_ref_vanilla/all \
    --csv datasets/coco_30k_10k.csv --device cuda:0
"""
import argparse, os, glob
import numpy as np
import torch
from PIL import Image
from scipy import linalg
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights
import open_clip


def list_images(d):
    fs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        fs.extend(glob.glob(os.path.join(d, ext)))
    return sorted(fs)


def case_of(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(name.split("_")[0])
    except ValueError:
        return None


class InceptionFeat:
    """2048-d pool3 features from torchvision InceptionV3 (FID feature space)."""
    def __init__(self, device):
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(device)
        self.device = device
        self.tf = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, paths, bs=32):
        out = []
        for i in range(0, len(paths), bs):
            batch = [self.tf(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]
            x = torch.stack(batch).to(self.device)
            out.append(self.net(x).cpu().numpy())
        return np.concatenate(out, 0)


def frechet_distance(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1 + s2 - 2.0 * covmean))


@torch.no_grad()
def clip_score(paths, case2prompt, device):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    sims = []
    bs = 32
    paths = [p for p in paths if case_of(p) in case2prompt]
    for i in range(0, len(paths), bs):
        chunk = paths[i:i+bs]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        texts = tokenizer([case2prompt[case_of(p)] for p in chunk]).to(device)
        ie = model.encode_image(imgs)
        te = model.encode_text(texts)
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        sims.append((ie * te).sum(-1).cpu().numpy())
    sims = np.concatenate(sims, 0) if sims else np.array([0.0])
    return float(sims.mean()), len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="dir of generated images to score")
    ap.add_argument("--ref_dir", default=None, help="reference image dir for FID (e.g. vanilla SD)")
    ap.add_argument("--csv", required=True, help="csv with case_number,prompt for CLIP score")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv)
    case2prompt = {int(r["case_number"]): str(r["prompt"]) for _, r in df.iterrows()}

    gen = list_images(args.gen_dir)
    assert gen, f"no images in {args.gen_dir}"

    cs, n_cs = clip_score(gen, case2prompt, args.device)

    fid = None
    if args.ref_dir:
        ref = list_images(args.ref_dir)
        assert ref, f"no images in {args.ref_dir}"
        feat = InceptionFeat(args.device)
        fid = frechet_distance(feat(gen), feat(ref))

    print(f"gen_dir   : {args.gen_dir}  ({len(gen)} imgs)")
    print(f"CLIP score: {cs:.4f}  (n={n_cs})")
    if fid is not None:
        print(f"FID vs ref: {fid:.3f}  (ref={args.ref_dir}, {len(ref)} imgs)")


if __name__ == "__main__":
    main()
