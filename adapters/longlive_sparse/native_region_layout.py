"""Physical organizations of one frozen ROI, with identical logical output."""
import hashlib
import json


def region_layout_plan(selected,layout,*,frames=8,height=22,width=40):
    selected=sorted(set(map(int,selected)));frame_tokens=height*width;total=frames*frame_tokens
    if min(frames,height,width)<1 or not selected or selected[0]<0 or selected[-1]>=total:
        raise ValueError('invalid source ROI')
    units=[]
    if layout in ('exact','page256','frame','block64'):
        unit_size={'exact':1,'page256':256,'frame':frame_tokens,'block64':64}[layout]
        ranges=[(f*frame_tokens,(f+1)*frame_tokens) for f in range(frames)] if layout=='block64' else [(0,total)]
        for start,end in ranges:
            for first in range(start,end,unit_size):units.append([i if i<end else -1 for i in range(first,first+unit_size)])
    elif layout in ('spatial4','spatial8'):
        edge=int(layout.removeprefix('spatial'));unit_size=edge*edge
        for frame in range(frames):
            for top in range(0,height,edge):
                for left in range(0,width,edge):
                    units.append([frame*frame_tokens+y*width+x if y<height and x<width else -1
                        for y in range(top,top+edge) for x in range(left,left+edge)])
    else:raise ValueError('unknown physical layout')
    physical_to_logical=[i for unit in units for i in unit]
    inverse={logical:physical for physical,logical in enumerate(physical_to_logical) if logical>=0}
    assert len(inverse)==total
    chosen=sorted({inverse[logical]//unit_size for logical in selected})
    packed_physical=[i for unit in chosen for i in range(unit*unit_size,(unit+1)*unit_size)]
    destination={physical:index for index,physical in enumerate(packed_physical)}
    gather=[destination[inverse[logical]] for logical in selected]
    runs=[]
    for unit in chosen:
        start=unit*unit_size;end=start+unit_size
        if runs and runs[-1][1]==start:runs[-1][1]=end
        else:runs.append([start,end,destination[start]])
    valid=sum(physical_to_logical[i]>=0 for i in packed_physical)
    return dict(layout=layout,logical_selected=selected,logical_coordinate_sha256=hashlib.sha256(json.dumps(selected,separators=(',',':')).encode()).hexdigest(),
        physical_to_logical=physical_to_logical,packed_physical_indices=packed_physical,gather_indices=gather,
        runs=runs,scheduled_tokens=len(packed_physical),valid_payload_tokens=valid,
        padding_tokens=len(packed_physical)-valid,archive_tokens=len(physical_to_logical))
