# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

from pkg_resources import parse_version
import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO
from enum import Enum

class OpenapiHeader(KaitaiStruct):

    class EMessageType(Enum):
        e_signal_data = 1
        e_data_quality = 2
        e_interpretation = 8
        e_aux_sequence_data = 11
            
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.magic = self._io.read_bytes(2)
        if not self.magic == b"\x42\x4B":
            raise kaitaistruct.ValidationNotEqualError(b"\x42\x4B", self.magic, self._io, u"/types/header/seq/0")
        self.header_length = self._io.read_u2le()
        self.message_type = KaitaiStream.resolve_enum(OpenapiMessage.Header.EMessageType, self._io.read_u2le())
        self.reserved1 = self._io.read_u2le()
        self.reserved2 = self._io.read_u4le()
        self.time = OpenapiMessage.Time(self._io, self, self._root)
        self.message_length = self._io.read_u4le()

if parse_version(kaitaistruct.__version__) < parse_version('0.9'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class OpenapiMessage(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.header = OpenapiMessage.Header(self._io, self, self._root)
        _on = self.header.message_type
        if _on == OpenapiMessage.Header.EMessageType.e_signal_data:
            self._raw_message = self._io.read_bytes(self.header.message_length)
            _io__raw_message = KaitaiStream(BytesIO(self._raw_message))
            self.message = OpenapiMessage.SignalData(_io__raw_message, self, self._root)
        elif _on == OpenapiMessage.Header.EMessageType.e_data_quality:
            self._raw_message = self._io.read_bytes(self.header.message_length)
            _io__raw_message = KaitaiStream(BytesIO(self._raw_message))
            self.message = OpenapiMessage.DataQuality(_io__raw_message, self, self._root)
        elif _on == OpenapiMessage.Header.EMessageType.e_interpretation:
            self._raw_message = self._io.read_bytes(self.header.message_length)
            _io__raw_message = KaitaiStream(BytesIO(self._raw_message))
            self.message = OpenapiMessage.Interpretations(_io__raw_message, self, self._root)
        else:
            self.message = self._io.read_bytes(self.header.message_length)

    class Interpretations(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.interpretations = []
            i = 0
            while not self._io.is_eof():
                self.interpretations.append(OpenapiMessage.Interpretation(self._io, self, self._root))
                i += 1



    class DataQuality(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.number_of_signals = self._io.read_u2le()
            self.qualities = [None] * (self.number_of_signals)
            for i in range(self.number_of_signals):
                self.qualities[i] = OpenapiMessage.DataQualityBlock(self._io, self, self._root)



    class DataQualityBlock(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.signal_id = self._io.read_u2le()
            self.validity_flags = OpenapiMessage.ValidityFlags(self._io, self, self._root)
            self.reserved = self._io.read_u2le()


    class Interpretation(KaitaiStruct):

        class EDescriptorType(Enum):
            data_type = 1
            scale_factor = 2
            offset = 3
            period_time = 4
            unit = 5
            vector_length = 6
            channel_type = 7
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.signal_id = self._io.read_u2le()
            self.descriptor_type = KaitaiStream.resolve_enum(OpenapiMessage.Interpretation.EDescriptorType, self._io.read_u2le())
            self.reserved = self._io.read_u2le()
            self.value_length = self._io.read_u2le()
            _on = self.descriptor_type
            if _on == OpenapiMessage.Interpretation.EDescriptorType.data_type:
                self.value = self._io.read_u2le()
            elif _on == OpenapiMessage.Interpretation.EDescriptorType.scale_factor:
                self.value = self._io.read_f8le()
            elif _on == OpenapiMessage.Interpretation.EDescriptorType.unit:
                self.value = OpenapiMessage.String(self._io, self, self._root)
            elif _on == OpenapiMessage.Interpretation.EDescriptorType.vector_length:
                self.value = self._io.read_u2le()
            elif _on == OpenapiMessage.Interpretation.EDescriptorType.period_time:
                self.value = OpenapiMessage.Time(self._io, self, self._root)
            elif _on == OpenapiMessage.Interpretation.EDescriptorType.offset:
                self.value = self._io.read_f8le()
            elif _on == OpenapiMessage.Interpretation.EDescriptorType.channel_type:
                self.value = self._io.read_u2le()
            self.padding = [None] * (((4 - (self._io.pos() % 4)) & 3))
            for i in range(((4 - (self._io.pos() % 4)) & 3)):
                self.padding[i] = self._io.read_u1()



    class String(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.count = self._io.read_u2le()
            self.data = (self._io.read_bytes(self.count)).decode(u"UTF-8")


    class TimeFamily(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.k = self._io.read_u1()
            self.l = self._io.read_u1()
            self.m = self._io.read_u1()
            self.n = self._io.read_u1()


    class ValidityFlags(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.f = self._io.read_u2le()

        @property
        def overload(self):
            if hasattr(self, '_m_overload'):
                return self._m_overload if hasattr(self, '_m_overload') else None

            self._m_overload = (self.f & 2) != 0
            return self._m_overload if hasattr(self, '_m_overload') else None

        @property
        def invalid(self):
            if hasattr(self, '_m_invalid'):
                return self._m_invalid if hasattr(self, '_m_invalid') else None

            self._m_invalid = (self.f & 8) != 0
            return self._m_invalid if hasattr(self, '_m_invalid') else None

        @property
        def overrun(self):
            if hasattr(self, '_m_overrun'):
                return self._m_overrun if hasattr(self, '_m_overrun') else None

            self._m_overrun = (self.f & 16) != 0
            return self._m_overrun if hasattr(self, '_m_overrun') else None


    class SignalData(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.number_of_signals = self._io.read_u2le()
            self.reserved = self._io.read_u2le()
            self.signals = [None] * (self.number_of_signals)
            for i in range(self.number_of_signals):
                self.signals[i] = OpenapiMessage.SignalBlock(self._io, self, self._root)



    class Header(KaitaiStruct):

        class EMessageType(Enum):
            e_signal_data = 1
            e_data_quality = 2
            e_interpretation = 8
            e_aux_sequence_data = 11
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.magic = self._io.read_bytes(2)
            if not self.magic == b"\x42\x4B":
                raise kaitaistruct.ValidationNotEqualError(b"\x42\x4B", self.magic, self._io, u"/types/header/seq/0")
            self.header_length = self._io.read_u2le()
            self.message_type = KaitaiStream.resolve_enum(OpenapiMessage.Header.EMessageType, self._io.read_u2le())
            self.reserved1 = self._io.read_u2le()
            self.reserved2 = self._io.read_u4le()
            self.time = OpenapiMessage.Time(self._io, self, self._root)
            self.message_length = self._io.read_u4le()


    class SignalBlock(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.signal_id = self._io.read_s2le()
            self.number_of_values = self._io.read_s2le()
            self.values = [None] * (self.number_of_values)
            for i in range(self.number_of_values):
                self.values[i] = OpenapiMessage.Value(self._io, self, self._root)



    class Time(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.time_family = OpenapiMessage.TimeFamily(self._io, self, self._root)
            self.time_count = self._io.read_u8le()


    class Value(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.value1 = self._io.read_u1()
            self.value2 = self._io.read_u1()
            self.value3 = self._io.read_s1()

        @property
        def calc_value(self):
            if hasattr(self, '_m_calc_value'):
                return self._m_calc_value if hasattr(self, '_m_calc_value') else None

            self._m_calc_value = ((self.value1 + (self.value2 << 8)) + (self.value3 << 16))
            return self._m_calc_value if hasattr(self, '_m_calc_value') else None