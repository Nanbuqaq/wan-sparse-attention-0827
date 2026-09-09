#!/usr/bin/env python3
"""Figure from already-audited native frames; always retain both seeds and tails."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from PIL import Image,ImageDraw,ImageFont
    p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from matplotlib.font_manager import findfont
    font_path=findfont('DejaVu Sans')
    def font(size):return ImageFont.truetype(font_path,size)
    refs=[]
    for seed in (20260925,20260926):
        old=args.metrics/'chest_causal_memory509_review_v2'
        text=args.metrics/'chest_past_text_control509_v1'/f'seed{seed}'
        hybrid=args.metrics/'chest_condition_history509_v1'/f'seed{seed}'
        source=old/f'seed{seed}__dense__native188.png'
        cells=[('Original return / no archive bank',lambda i:old/f'seed{seed}__dense__native{i}.png'),
               ('Original return / recent KV',lambda i:old/f'seed{seed}__recent_virtual__native{i}.png'),
               ('Past-state text / no archive bank',lambda i:text/f'past_text__native{i}.png'),
               ('Past-state text / recent KV',lambda i:hybrid/f'text_and_history__native{i}.png')]
        board=Image.new('RGB',(1280,840),'#f8fafc');draw=ImageDraw.Draw(board)
        draw.text((24,16),f'Current state condition x raw history | seed {seed}',font=font(25),fill='#142f43')
        def paste(path,xy,size):
            with Image.open(path) as image:board.paste(image.convert('RGB').resize(size),xy)
            refs.append(dict(seed=seed,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        paste(source,(24,64),(384,211));draw.text((436,80),'Completed source: pixel 188',font=font(23),fill='#142f43')
        draw.text((436,124),'Open lid; blue cloth inside.',font=font(21),fill='#304b61')
        draw.text((436,167),'Every cell shares the actual pre-return prefix.',font=font(19),fill='#304b61')
        draw.text((436,205),'KV cells retrieve the same past 8-latent source.',font=font(19),fill='#304b61')
        for index,(name,resolve) in enumerate(cells):
            x,y=24+(index%2)*640,302+(index//2)*242
            draw.text((x,y),name,font=font(20),fill='#142f43')
            for col,pixel in enumerate((412,508)):
                draw.text((x+col*300,y+30),f'pixel {pixel}',font=font(16),fill='#486073')
                paste(resolve(pixel),(x+col*300,y+54),(288,158))
        note='Seed 25 still closes late. Source-detail recovery is not reliable long-term state retention.'
        if seed==20260926:note='Seed 26 holds the broad open state in sampled tails; this does not establish general success.'
        draw.text((24,792),note,font=font(16),fill='#684339')
        draw.text((24,815),'Structured past-text diagnostic; descriptive examples, not autonomous memory or SOTA scoring.',font=font(15),fill='#486073')
        board.save(args.output/f'seed{seed}.png')
    (args.output/'sources.json').write_text(json.dumps(dict(source_frames=refs,both_seeds_retained=True,
        no_new_video_generated=True,no_blind_or_quantitative_quality_claim=True),indent=2)+'\n')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>State condition and KV factorial</title>'
        '<style>body{font:18px system-ui;max-width:1320px;margin:25px auto}img{width:100%}</style>'
        '<h1>状态描述与历史KV：两seed的完整2×2例图</h1><p>完整视频已审计相同干预前prefix。图保留两seed和尾帧；'
        '组合更接近源外观，但seed25后段仍合盖，不能只看早段宣布状态保持成功。</p>'+
        ''.join(f'<h2>{seed}</h2><img src="seed{seed}.png">' for seed in (20260925,20260926)))


if __name__=='__main__':main()
