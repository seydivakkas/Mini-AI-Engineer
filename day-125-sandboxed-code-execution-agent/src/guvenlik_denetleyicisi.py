"""
AST Tabanlı Statik Güvenlik Denetleyicisi Modülü (Day 125 - Faz 7).
Çalıştırılmak istenen Python kodunun soyut sözdizim ağacını (AST) analiz ederek zararlı sistem çağrılarını ve kaçış girişimlerini engeller.
"""

import ast
from typing import Tuple, List, Set


class AstGuvenlikDenetleyicisi:
    """Python AST ağacını tarayarak güvenlik ihlallerini tespit eden denetleyici."""

    # Yasaklı Kütüphaneler (Sistem, İşlem, Ağ ve Bellek)
    YASAKLI_MODULLER: Set[str] = {
        "os", "sys", "subprocess", "shutil", "socket", "ctypes",
        "builtins", "posix", "nt", "pty", "threading", "multiprocessing",
        "pickle", "shelve", "dbm", "sqlite3", "requests", "urllib", "http",
    }

    # Yasaklı Fonksiyon Çağrıları (Dinamik Kod İcrası ve Dosya/Sistem İşlemleri)
    YASAKLI_FONKSIYONLAR: Set[str] = {
        "eval", "exec", "open", "compile", "__import__", "globals",
        "locals", "getattr", "setattr", "delattr", "system", "popen",
        "spawn", "fork", "kill", "exit", "quit",
    }

    # Yasaklı Nitelikler (Sandbox Escape / Kaçış Vektörleri)
    YASAKLI_NITELIKLER: Set[str] = {
        "__subclasses__", "__bases__", "__mro__", "__globals__",
        "__code__", "__closure__", "__builtins__", "__dict__",
    }

    @classmethod
    def denetle(cls, kod_metni: str) -> Tuple[bool, List[str], float]:
        """
        Kod metnini AST üzerinden statik analiz eder.
        Dönüş: (guvenli_mi: bool, ihlaller: List[str], guvenlik_skoru: float)
        """
        ihlaller: List[str] = []

        try:
            agac = ast.parse(kod_metni)
        except SyntaxError as e:
            return False, [f"Sözdizim Hatası (SyntaxError): {str(e)}"], 0.0

        for dugum in ast.walk(agac):
            # 1. Modül Import Denetimi (import os, from subprocess import Popen)
            if isinstance(dugum, ast.Import):
                for isim in dugum.names:
                    ana_modul = isim.name.split(".")[0]
                    if ana_modul in cls.YASAKLI_MODULLER:
                        ihlaller.append(f"Yasaklı kütüphane importu tespit edildi: '{isim.name}' (Satır: {dugum.lineno})")

            elif isinstance(dugum, ast.ImportFrom):
                if dugum.module:
                    ana_modul = dugum.module.split(".")[0]
                    if ana_modul in cls.YASAKLI_MODULLER:
                        ihlaller.append(f"Yasaklı kütüphaneden import tespit edildi: '{dugum.module}' (Satır: {dugum.lineno})")

            # 2. Tehlikeli Fonksiyon Çağrıları (eval(), open(), exec())
            elif isinstance(dugum, ast.Call):
                if isinstance(dugum.func, ast.Name):
                    if dugum.func.id in cls.YASAKLI_FONKSIYONLAR:
                        ihlaller.append(f"Yasaklı sistem/yerleşik fonksiyon çağrısı: '{dugum.func.id}()' (Satır: {dugum.lineno})")

            # 3. Sandbox Escape ve Özel Nitelik Erişimi (__subclasses__, __globals__)
            elif isinstance(dugum, ast.Attribute):
                if dugum.attr in cls.YASAKLI_NITELIKLER:
                    ihlaller.append(f"Güvenlik kaçış vektörü (Özel Nitelik) erişimi: '{dugum.attr}' (Satır: {dugum.lineno})")

        guvenli_mi = len(ihlaller) == 0
        guvenlik_skoru = 100.0 if guvenli_mi else max(0.0, 100.0 - len(ihlaller) * 35.0)

        return guvenli_mi, ihlaller, guvenlik_skoru
