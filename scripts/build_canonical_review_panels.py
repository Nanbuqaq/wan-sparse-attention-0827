#!/usr/bin/env python3
"""All four quarters side-by-side; identical latent systems share one column."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw


def retile_quarter(image):
    """Rearrange 8x2 thumbnails as 4x4 without stretching any pixel."""
    if image.size != (1664, 240):
        raise ValueError('expected sixteen208x120 thumbnails in an8x2 board')
    result = Image.new('RGB', (832, 480))
    for slot in range(16):
        x, y = (slot%8)*208, (slot//8)*120
        result.paste(image.crop((x, y, x+208, y+120)), ((slot%4)*208, (slot//4)*120))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quality-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    paths = sorted(args.quality_root.glob('lane*/*/quality.json'))
    if not paths:
        raise ValueError('no completed canonical groups')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for path in paths:
        quality = json.loads(path.read_text())
        if quality.get('metric_input_protocol_version') != 2 or quality['status'] != 'pass':
            raise ValueError('valid canonical v2 results required')
        columns = {}
        for row in quality['rows']:
            if row['status'] != 'pass':
                continue
            digest = row['latent_sha256']
            item = columns.setdefault(digest, {'cases': [], 'configs': [], 'render': row['canonical_render']})
            item['cases'].append(row['case_id'])
            item['configs'].append(row['formal_config_id'] or row['method'])
        width, height = 832, 480
        canvas = Image.new('RGB', (width*len(columns), 4*(height+28)+50), 'white')
        draw = ImageDraw.Draw(canvas)
        title = f"{quality['prompt']} seed={quality['seed']} latent={quality['latent_frames']} - all 4 quarters"
        draw.text((8, 8), title, fill='black')
        sources = []
        for column, item in enumerate(columns.values()):
            directory = Path(item['render']['render_directory'])
            for quarter in range(1, 5):
                board = directory/f'quarter{quarter}.png'
                with Image.open(board) as image:
                    resized = retile_quarter(image.convert('RGB'))
                    top = 50+(quarter-1)*(height+28)
                    canvas.paste(resized, (column*width, top+28))
                draw.text((column*width+8, top+6), f"Q{quarter}: {', '.join(item['configs'])}", fill='black')
                sources.append({'path': str(board.resolve()), 'sha256': hashlib.sha256(board.read_bytes()).hexdigest()})
        name = path.parent.name
        output = args.output/f'{name}.png'
        canvas.save(output)
        records.append({'group': name, 'prompt': quality['prompt'], 'seed': quality['seed'],
            'latent_frames': quality['latent_frames'], 'panel': str(output.resolve()),
            'panel_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'quality_path': str(path.resolve()), 'quality_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'columns': list(columns.values()), 'sources': sources,
            'review_status': 'awaiting_actual_visual_inspection_not_auto_scored'})
    (args.output/'index.json').write_text(json.dumps({'status': 'pass', 'groups': records,
        'all_four_quarters_included': True, 'thumbnail_aspect_ratio_preserved': True,
        'panels_are_samples_not_full_video_review': True}, indent=2)+'\n')
    print(json.dumps({'status': 'pass', 'groups': len(records), 'quarter_columns': sum(len(r['columns'])*4 for r in records)}))


if __name__ == '__main__':
    main()
