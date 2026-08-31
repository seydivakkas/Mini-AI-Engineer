"""
Tesla UDS (Unified Diagnostic Services - ISO 14229) & OBD-II Protokol Modulu
============================================================================
Bu modul; ISO 14229-1 (UDS), ISO 15031-6 (OBD-II DTCs) ve ISO 15765-2 (ISO-TP)
standartlarinda otomotiv teshis sunucusu ve istemcisi gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import struct
import time
import hashlib


class UDSServiceID(IntEnum):
    DIAGNOSTIC_SESSION_CONTROL = 0x10
    ECU_RESET = 0x11
    CLEAR_DIAGNOSTIC_INFORMATION = 0x14
    READ_DTC_INFORMATION = 0x19
    READ_DATA_BY_IDENTIFIER = 0x22
    SECURITY_ACCESS = 0x27
    WRITE_DATA_BY_IDENTIFIER = 0x2E
    TESTER_PRESENT = 0x3E
    NEGATIVE_RESPONSE = 0x7F


class UDSNRC(IntEnum):
    """Negative Response Codes (Negatif Yanıt Kodları - ISO 14229-1)"""
    POSITIVE_RESPONSE = 0x00
    GENERAL_REJECT = 0x10
    SERVICE_NOT_SUPPORTED = 0x11
    SUB_FUNCTION_NOT_SUPPORTED = 0x12
    INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT = 0x13
    CONDITIONS_NOT_CORRECT = 0x22
    REQUEST_OUT_OF_RANGE = 0x31
    SECURITY_ACCESS_DENIED = 0x33
    INVALID_KEY = 0x35
    EXCEEDED_NUMBER_OF_ATTEMPTS = 0x36
    REQUIRED_TIME_DELAY_NOT_EXPIRED = 0x37


class DiagnosticSessionType(IntEnum):
    DEFAULT_SESSION = 0x01
    PROGRAMMING_SESSION = 0x02
    EXTENDED_DIAGNOSTIC_SESSION = 0x03
    SAFETY_SYSTEM_DIAGNOSTIC_SESSION = 0x04


@dataclass
class DTCRecord:
    """
    3-Baytlık ISO 14229 / ISO 15031 DTC (Diagnostic Trouble Code) Nesnesi.
    Örnek: P0A1F-00 -> Battery Energy Control Module Performance Fault
    """
    dtc_bytes: bytes      # 3 Bayt (High, Middle, Low/FTB)
    status_mask: int      # 8-bit durum bayrağı (bit 3: Confirmed, bit 0: Active)
    aciklama: str = ""

    @property
    def formatted_dtc(self) -> str:
        """3 baytlık ham DTC verisinden standart P/C/B/U kodunu türetir."""
        b1, b2, b3 = self.dtc_bytes
        kategori_kodu = (b1 >> 6) & 0x03
        kategori = ["P", "C", "B", "U"][kategori_kodu]
        d1 = (b1 >> 4) & 0x03
        d2 = b1 & 0x0F
        d3 = (b2 >> 4) & 0x0F
        d4 = b2 & 0x0F
        return f"{kategori}{d1}{d2:X}{d3:X}{d4:X}-{b3:02X}"

    @property
    def is_confirmed(self) -> bool:
        return bool(self.status_mask & 0x08)

    @property
    def is_active(self) -> bool:
        return bool(self.status_mask & 0x01)


def decode_dtc(raw_bytes: bytes) -> str:
    """
    3 baytlık ham DTC verisini standart koda dönüştürür.
    Örnek: bytes([0x0A, 0x1F, 0x16]) -> 'P0A1F-16'
    """
    if len(raw_bytes) < 3:
        raise ValueError("DTC en az 3 bayt olmalıdır.")
    b1, b2, b3 = raw_bytes[:3]
    category = ["P", "C", "B", "U"][(b1 >> 6) & 0x03]
    d1 = (b1 >> 4) & 0x03
    d2 = b1 & 0x0F
    d3 = (b2 >> 4) & 0x0F
    d4 = b2 & 0x0F
    return f"{category}{d1}{d2:X}{d3:X}{d4:X}-{b3:02X}"


def encode_dtc(category: str, code_int: int, fault_type: int = 0x00) -> bytes:
    """
    Kategori ve koddan 3 baytlık DTC bayt dizisi üretir.
    Örnek: encode_dtc('P', 0x0A1F, 0x00)
    """
    cat_map = {"P": 0, "C": 1, "B": 2, "U": 3}
    cat_val = cat_map.get(category.upper(), 0)
    b1 = (cat_val << 6) | ((code_int >> 8) & 0x3F)
    b2 = code_int & 0xFF
    b3 = fault_type & 0xFF
    return bytes([b1, b2, b3])


class TeslaUDSServer:
    """
    Tesla Model S/3/X/Y Araç İçi UDS Teşhis Sunucusu (ECU).
    Desteklenen Servisler:
      - 0x10: DiagnosticSessionControl
      - 0x14: ClearDiagnosticInformation
      - 0x19: ReadDTCInformation
      - 0x22: ReadDataByIdentifier (DID)
      - 0x27: SecurityAccess (Seed-Key Doğrulama)
      - 0x2E: WriteDataByIdentifier (DID)
      - 0x3E: TesterPresent
    """
    def __init__(self, ecu_name: str = "Tesla BMS Core ECU"):
        self.ecu_name = ecu_name
        self.current_session = DiagnosticSessionType.DEFAULT_SESSION
        self.security_unlocked = False
        self.current_seed: Optional[bytes] = None
        self.security_secret_key = b"TeslaSuperSecretSecKey2026"
        self.failed_security_attempts = 0

        # Veri Tanımlayıcıları (DID - Data Identifiers)
        self.did_database: Dict[int, bytes] = {
            0xF190: b"5YJ3E1EB8NF123456",                      # VIN (Vehicle Identification Number)
            0xF189: b"v12.5.4-fsd-2026.8",                     # Software Version
            0x0100: struct.pack(">f", 398.6),                   # Pack Voltage (V)
            0x0101: struct.pack(">f", 32.4),                    # Inverter Temp (°C)
            0x0102: struct.pack(">i", 6450),                    # Motor RPM
            0x0103: bytes([0x01]),                              # FSD Autopilot State (1: Engaged)
            0x0200: struct.pack(">f", 78.5)                     # Battery SoC (%)
        }

        # Kayıtlı DTC Listesi
        self.dtc_database: Dict[str, DTCRecord] = {
            "P0A1F-00": DTCRecord(
                dtc_bytes=encode_dtc("P", 0x0A1F, 0x00),
                status_mask=0x2F,  # Confirmed & Active
                aciklama="Battery Energy Control Module Performance Fault"
            ),
            "U0100-87": DTCRecord(
                dtc_bytes=encode_dtc("U", 0x0100, 0x87),
                status_mask=0x08,  # Confirmed only
                aciklama="Lost Communication With ECM/PCM"
            ),
            "C1A00-14": DTCRecord(
                dtc_bytes=encode_dtc("C", 0x1A00, 0x14),
                status_mask=0x29,  # Active
                aciklama="Steering Angle Sensor Circuit Ground Short"
            )
        }

    def _generate_security_key(self, seed: bytes) -> bytes:
        """Seed üzerinden SHA256 tabanlı OEM yanıt anahtarı hesaplar."""
        return hashlib.sha256(seed + self.security_secret_key).digest()[:4]

    def process_request(self, request_payload: bytes) -> bytes:
        """Gelen UDS istek paketini işler ve UDS yanıt paketi döner."""
        if not request_payload:
            return bytes([UDSServiceID.NEGATIVE_RESPONSE, 0x00, UDSNRC.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT])

        sid = request_payload[0]

        # 1. 0x10: DiagnosticSessionControl
        if sid == UDSServiceID.DIAGNOSTIC_SESSION_CONTROL:
            if len(request_payload) < 2:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT])
            session_type = request_payload[1]
            try:
                self.current_session = DiagnosticSessionType(session_type)
                # Oturum değiştiğinde güvenlik kilitlenir
                self.security_unlocked = False
                # Pozitif yanıt: SID + 0x40, session_type, P2_server_max(50ms), P2*_server_max(5000ms)
                return bytes([sid + 0x40, session_type, 0x00, 0x32, 0x01, 0xF4])
            except ValueError:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.SUB_FUNCTION_NOT_SUPPORTED])

        # 2. 0x3E: TesterPresent
        elif sid == UDSServiceID.TESTER_PRESENT:
            if len(request_payload) < 2:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT])
            sub_func = request_payload[1]
            return bytes([sid + 0x40, sub_func & 0x7F])

        # 3. 0x22: ReadDataByIdentifier
        elif sid == UDSServiceID.READ_DATA_BY_IDENTIFIER:
            if len(request_payload) < 3:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT])
            did = struct.unpack(">H", request_payload[1:3])[0]
            if did not in self.did_database:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.REQUEST_OUT_OF_RANGE])
            
            did_data = self.did_database[did]
            return bytes([sid + 0x40]) + struct.pack(">H", did) + did_data

        # 4. 0x27: SecurityAccess (Seed-Key)
        elif sid == UDSServiceID.SECURITY_ACCESS:
            if len(request_payload) < 2:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT])
            sub_func = request_payload[1]

            # 0x01: Request Seed
            if sub_func == 0x01:
                if self.security_unlocked:
                    # Zaten açık ise 0x00 seed döner
                    return bytes([sid + 0x40, 0x01, 0x00, 0x00, 0x00, 0x00])
                # Yeni 4 baytlık seed üret
                self.current_seed = bytes([0xA5, 0x5A, 0x3C, 0xC3])
                return bytes([sid + 0x40, 0x01]) + self.current_seed

            # 0x02: Send Key
            elif sub_func == 0x02:
                if len(request_payload) < 6:
                    return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT])
                if not self.current_seed:
                    return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.CONDITIONS_NOT_CORRECT])

                sent_key = request_payload[2:6]
                expected_key = self._generate_security_key(self.current_seed)

                if sent_key == expected_key:
                    self.security_unlocked = True
                    self.failed_security_attempts = 0
                    self.current_seed = None
                    return bytes([sid + 0x40, 0x02])
                else:
                    self.failed_security_attempts += 1
                    if self.failed_security_attempts >= 3:
                        return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.EXCEEDED_NUMBER_OF_ATTEMPTS])
                    return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.INVALID_KEY])

            else:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.SUB_FUNCTION_NOT_SUPPORTED])

        # 5. 0x2E: WriteDataByIdentifier
        elif sid == UDSServiceID.WRITE_DATA_BY_IDENTIFIER:
            if len(request_payload) < 4:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT])
            
            # Yazma işlemi için Extended Session ve Güvenlik Kilidi açık olmalıdır
            if self.current_session == DiagnosticSessionType.DEFAULT_SESSION:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.CONDITIONS_NOT_CORRECT])
            if not self.security_unlocked:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.SECURITY_ACCESS_DENIED])

            did = struct.unpack(">H", request_payload[1:3])[0]
            new_data = request_payload[3:]
            self.did_database[did] = new_data
            return bytes([sid + 0x40]) + struct.pack(">H", did)

        # 6. 0x19: ReadDTCInformation
        elif sid == UDSServiceID.READ_DTC_INFORMATION:
            if len(request_payload) < 2:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT])
            sub_func = request_payload[1]

            # 0x02: reportDTCByStatusMask
            if sub_func == 0x02:
                mask = request_payload[2] if len(request_payload) >= 3 else 0xFF
                dtc_bytes_list = bytearray()
                for rec in self.dtc_database.values():
                    if rec.status_mask & mask:
                        dtc_bytes_list.extend(rec.dtc_bytes)
                        dtc_bytes_list.append(rec.status_mask)
                
                # Pozitif Yanıt: 0x59 0x02 statusAvailabilityMask (0xFF) + [DTC_High, DTC_Mid, DTC_Low, Status]
                return bytes([sid + 0x40, 0x02, 0xFF]) + bytes(dtc_bytes_list)

            # 0x01: reportNumberOfDTCByStatusMask
            elif sub_func == 0x01:
                mask = request_payload[2] if len(request_payload) >= 3 else 0xFF
                count = sum(1 for rec in self.dtc_database.values() if rec.status_mask & mask)
                # 0x59 0x01 statusAvailabilityMask (0xFF) formatIdentifier(0x00=ISO14229-1) dtcCount(2-byte)
                return bytes([sid + 0x40, 0x01, 0xFF, 0x00]) + struct.pack(">H", count)

            else:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.SUB_FUNCTION_NOT_SUPPORTED])

        # 7. 0x14: ClearDiagnosticInformation
        elif sid == UDSServiceID.CLEAR_DIAGNOSTIC_INFORMATION:
            if len(request_payload) < 4:
                return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT])
            # 0xFFFFFF: Clear all DTCs
            group = request_payload[1:4]
            if group == b"\xFF\xFF\xFF":
                self.dtc_database.clear()
                return bytes([sid + 0x40])
            return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.REQUEST_OUT_OF_RANGE])

        # Desteklenmeyen Servis
        return bytes([UDSServiceID.NEGATIVE_RESPONSE, sid, UDSNRC.SERVICE_NOT_SUPPORTED])


class TeslaUDSClient:
    """
    Tesla Teşhis Cihazı (Service Tool / Gateway Tester).
    """
    def __init__(self, server: TeslaUDSServer):
        self.server = server

    def set_session(self, session: DiagnosticSessionType) -> Tuple[bool, DiagnosticSessionType]:
        req = bytes([UDSServiceID.DIAGNOSTIC_SESSION_CONTROL, int(session)])
        resp = self.server.process_request(req)
        if resp[0] == UDSServiceID.DIAGNOSTIC_SESSION_CONTROL + 0x40:
            return True, DiagnosticSessionType(resp[1])
        return False, session

    def unlock_security(self) -> bool:
        # 1. Request Seed
        req_seed = bytes([UDSServiceID.SECURITY_ACCESS, 0x01])
        resp_seed = self.server.process_request(req_seed)
        if resp_seed[0] != UDSServiceID.SECURITY_ACCESS + 0x40 or len(resp_seed) < 6:
            return False

        seed = resp_seed[2:6]
        if seed == bytes(4):
            return True  # Zaten açık

        # 2. Compute Key & Send Key
        key = self.server._generate_security_key(seed)
        req_key = bytes([UDSServiceID.SECURITY_ACCESS, 0x02]) + key
        resp_key = self.server.process_request(req_key)
        return resp_key[0] == UDSServiceID.SECURITY_ACCESS + 0x40

    def read_did(self, did: int) -> Tuple[bool, Optional[bytes]]:
        req = bytes([UDSServiceID.READ_DATA_BY_IDENTIFIER]) + struct.pack(">H", did)
        resp = self.server.process_request(req)
        if resp[0] == UDSServiceID.READ_DATA_BY_IDENTIFIER + 0x40:
            return True, resp[3:]
        return False, None

    def write_did(self, did: int, data: bytes) -> bool:
        req = bytes([UDSServiceID.WRITE_DATA_BY_IDENTIFIER]) + struct.pack(">H", did) + data
        resp = self.server.process_request(req)
        return resp[0] == UDSServiceID.WRITE_DATA_BY_IDENTIFIER + 0x40

    def read_dtcs(self, status_mask: int = 0xFF) -> List[Tuple[str, int]]:
        req = bytes([UDSServiceID.READ_DTC_INFORMATION, 0x02, status_mask])
        resp = self.server.process_request(req)
        dtcs = []
        if resp[0] == UDSServiceID.READ_DTC_INFORMATION + 0x40 and len(resp) >= 3:
            raw_dtcs = resp[3:]
            # Her DTC kaydı 4 bayttır: [DTC_High, DTC_Mid, DTC_Low, StatusMask]
            for i in range(0, len(raw_dtcs), 4):
                if i + 4 <= len(raw_dtcs):
                    dtc_chunk = raw_dtcs[i:i+3]
                    status = raw_dtcs[i+3]
                    formatted = decode_dtc(dtc_chunk)
                    dtcs.append((formatted, status))
        return dtcs

    def clear_dtcs(self) -> bool:
        req = bytes([UDSServiceID.CLEAR_DIAGNOSTIC_INFORMATION, 0xFF, 0xFF, 0xFF])
        resp = self.server.process_request(req)
        return resp[0] == UDSServiceID.CLEAR_DIAGNOSTIC_INFORMATION + 0x40
