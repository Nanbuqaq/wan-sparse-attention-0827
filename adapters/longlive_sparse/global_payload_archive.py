"""Compose a qualified read-only global payload control with archive execution."""
from .staged_scene_archive import ArchiveStagingMixin
from .global_payload_control import GlobalPayloadMixin
from .immutable_source_reader import ImmutableSourceReader
from .layer_stream_source_reader import LayerStreamSourceReader


class ArchivedGlobalPayloadResident(ArchiveStagingMixin,GlobalPayloadMixin,ImmutableSourceReader):pass
class ArchivedGlobalPayloadStream(ArchiveStagingMixin,GlobalPayloadMixin,LayerStreamSourceReader):pass
