#!/usr/bin/env python3
"""Summarize the audited two-by-two mask-reuse/fill development comparison."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=Path, required=True, help='JSON mapping of four named case/audit directories')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cases = json.loads(args.cases.read_text())
    keys = ('all32_uniform', 'all32_stable', 'holdfirst_uniform', 'holdfirst_stable')
    assert set(cases) == set(keys)
    audits, latent = {}, {}
    contact = Image.new('RGB', (1920, 224 * 4), 'white')
    draw = ImageDraw.Draw(contact)
    for row, key in enumerate(keys):
        case, audit = Path(cases[key]['case']), Path(cases[key]['audit'])
        audits[key] = json.loads((audit / 'audit.json').read_text())
        assert audits[key]['status'] == 'pass'
        latent[key] = torch.load(case / 'latents.pt', weights_only=True, map_location='cpu')[:, 96:].float()
        contact.paste(Image.open(audit / 'contact.jpg'), (0, row * 224 + 24))
        draw.text((4, row * 224 + 4), key, fill='black')
    pairs = {}
    for fill in ('uniform', 'stable'):
        before, after = latent['all32_' + fill], latent['holdfirst_' + fill]
        difference = after - before
        pairs[fill] = dict(postreturn_latent_relative_L2=float(difference.norm() / before.norm()),
            postreturn_latent_MSE=float(difference.square().mean()),
            postreturn_latent_max_abs=float(difference.abs().max()),
            scope='perturbation magnitude, not a quality metric or teacher error')
    contact.save(args.output / 'factorial_contact.jpg', quality=95)
    result = dict(status='pass', cases=audits, temporal_reuse_pairs=pairs,
        quality='requires visual review; no method promotion from these numbers',
        scope='single toy13 development source, precomputed masks; not online latency')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(pairs))


if __name__ == '__main__':
    main()
