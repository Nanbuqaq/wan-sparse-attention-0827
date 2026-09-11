"""Bounded raw RGB windows tied to actual closed-scene and latent events."""
from collections import OrderedDict
import hashlib
import threading
import time

import torch


class NativeSourcePixelWitness:
    def __init__(self,*,started,byte_budget=384*1024**2,max_archives=3,on_ready=None):
        self.started=started;self.budget=byte_budget;self.max_archives=max_archives
        self.lock=threading.Lock();self.ring=OrderedDict();self.latents=OrderedDict()
        self.pending={};self.completed={};self.next_pixel=0;self.next_latent=0
        self.peak_bytes=0;self.copied_bytes=0;self.callback_host_s=0.
        self.last_archive_end=0;self.rgb_shape=None
        self.scene=None;self.original_archive=None
        self.on_ready=on_ready

    def _bytes(self):
        return sum(x.numel()*x.element_size() for x in self.ring.values())+sum(
            r['pixels'].numel()*r['pixels'].element_size() for r in self.completed.values())

    def _check_budget(self,extra=0):
        size=self._bytes()+extra
        if size>self.budget:raise RuntimeError('raw source witness CPU byte budget exceeded')
        self.peak_bytes=max(self.peak_bytes,size)

    def _complete_ready(self):
        for version,row in list(self.pending.items()):
            start,end=row['source_start'],row['source_end'];key=(start,end)
            a,b=row['pixel_start'],row['pixel_end']
            if key not in self.latents or not all(i in self.ring for i in range(a,b)):continue
            needed=sum(self.ring[i].numel()*self.ring[i].element_size() for i in range(a,b))
            self._check_budget(needed)
            pixels=torch.stack([self.ring[i] for i in range(a,b)])
            self.completed[version]=dict(row,pixels=pixels,source_latent_sha256=self.latents[key],
                raw_pixel_bytes_sha256=hashlib.sha256(memoryview(pixels.numpy())).hexdigest(),
                ready_s=time.perf_counter()-self.started)
            del self.pending[version]
            if self.on_ready is not None:self.on_ready(self.completed[version])

    def register(self,version,source_end,source_phase):
        began=time.perf_counter()
        with self.lock:
            if version in self.pending or version in self.completed or source_end<8 or source_end%8 or source_end<=self.last_archive_end:
                raise ValueError('unique whole closed-scene source required')
            if len(self.pending)+len(self.completed)>=self.max_archives:
                raise RuntimeError('diagnostic archive-count cap exceeded')
            start=source_end-8;a=max(0,4*start-3);b=4*source_end-3
            if self.ring and a<next(iter(self.ring)):
                raise RuntimeError('closed source pixels have already left the bounded ring')
            self.pending[version]=dict(archive_version=version,source_start=start,source_end=source_end,
                source_phase=source_phase,pixel_start=a,pixel_end=b,
                registered_s=time.perf_counter()-self.started)
            self.last_archive_end=source_end
            self._complete_ready()
            self.callback_host_s+=time.perf_counter()-began

    def on_latent(self,start,frames,digest):
        began=time.perf_counter()
        with self.lock:
            if start!=self.next_latent or frames!=8:raise ValueError('ordered committed eight-frame chunks required')
            self.next_latent+=frames;self.latents[(start,start+frames)]=digest
            while len(self.latents)>3:self.latents.popitem(last=False)
            self._complete_ready()
            self.callback_host_s+=time.perf_counter()-began

    def on_rgb(self,index,value):
        began=time.perf_counter()
        if value.device.type!='cpu' or value.dtype!=torch.uint8 or value.ndim!=3 or value.shape[-1]!=3:
            raise ValueError('canonical CPU RGB required without an implicit GPU copy')
        with self.lock:
            if index!=self.next_pixel:raise ValueError('raw RGB stream order changed')
            if self.rgb_shape is not None and tuple(value.shape)!=self.rgb_shape:raise ValueError('RGB geometry changed')
            self.rgb_shape=tuple(value.shape)
            self._check_budget(value.numel()*value.element_size())
            self.ring[index]=value.clone();self.copied_bytes+=value.numel()*value.element_size();self.next_pixel+=1
            while len(self.ring)>32:self.ring.popitem(last=False)
            self._complete_ready()
            self.callback_host_s+=time.perf_counter()-began

    def attach_scene(self,scene):
        self.scene=scene;self.original_archive=scene._archive_last_scene
        def archive(frame):
            self.original_archive(frame)
            d=scene.banks[-1]['descriptor']
            self.register(d.archive_version,d.source_end,d.source_phase)
        scene._archive_last_scene=archive

    def detach(self):
        if self.scene is not None:self.scene._archive_last_scene=self.original_archive
        self.scene=self.original_archive=None

    def export(self,path,expected_archives,allow_partial=False):
        with self.lock:
            complete=not self.pending and len(self.completed)==expected_archives
            if not complete and not allow_partial:
                raise RuntimeError('incomplete raw source/latent witness')
            records=[self.completed[k] for k in sorted(self.completed)]
            with path.open('xb') as f:torch.save(dict(schema='native_source_raw_rgb_v1',records=records,
                pixels_before_lossy_codec=True,offline_diagnostic_only=True,complete=complete,
                pending=list(self.pending.values())),f)
            return dict(complete=complete,records=[{k:v for k,v in r.items() if k!='pixels'} for r in records],
                CPU_owned_peak_bytes=self.peak_bytes,CPU_byte_budget=self.budget,
                CPU_budget_scope='owned RGB tensors; Python metadata and artifact serialization excluded',
                RGB_ring_copy_bytes=self.copied_bytes,callback_host_s=self.callback_host_s,
                bounded_ring_frames=32,max_archives=self.max_archives,
                source_events_from_actual_scene_archive=True,raw_RGB_before_lossy_codec=True,
                callback_times_include_lock_wait_and_can_overlap=True,
                feeds_live_consumer=self.on_ready is not None,
                diagnostic_only_not_online_geometry_producer=self.on_ready is None)
