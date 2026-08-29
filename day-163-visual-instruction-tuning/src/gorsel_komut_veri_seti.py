"""
Görsel Komut Veri Seti (Visual Instruction Dataset) Modülü (Day 163 - FAZ 9).
LLaVA-Instruct formatında VQA, Detaylı Açıklama ve Çok Adımlı Mantık verisi üretir.
"""

from typing import List, Dict, Any
import torch


class GorselKomutVeriSeti:
    """LLaVA-Instruct-150k Tarzı Sentetik Görsel Komut Veri Seti."""

    @classmethod
    def ornek_verileri_getir(cls) -> List[Dict[str, Any]]:
        return [
            {
                "id": "vqa_001",
                "kategori": "Kısa VQA",
                "goruntu_adi": "mutfak_masasi.jpg",
                "diyalog": [
                    {"rol": "insan", "metin": "<image>\nMasanın üzerinde hangi meyveler var?"},
                    {"rol": "asistan", "metin": "Masanın üzerinde iki adet kırmızı elma ve bir muz bulunmaktadır."},
                ],
                "prompt_token_uzunlugu": 14,
                "yanit_token_uzunlugu": 18,
            },
            {
                "id": "desc_002",
                "kategori": "Detaylı Açıklama",
                "goruntu_adi": "sehir_parki.jpg",
                "diyalog": [
                    {"rol": "insan", "metin": "<image>\nBu görseldeki sahneyi ayrıntılı olarak betimleyin."},
                    {"rol": "asistan", "metin": "Görselde güneşli bir günde yeşil bir şehir parkı görülmektedir. Arka planda yüksek binalar yükselirken, ön planda bankta oturan bir kişi kitap okumaktadır."},
                ],
                "prompt_token_uzunlugu": 12,
                "yanit_token_uzunlugu": 32,
            },
            {
                "id": "reasoning_003",
                "kategori": "Karmaşık Muhakeme",
                "goruntu_adi": "trafik_isigi.jpg",
                "diyalog": [
                    {"rol": "insan", "metin": "<image>\nSürücü neden yavaşlamalıdır?"},
                    {"rol": "asistan", "metin": "Önündeki trafik ışığı sarıdan kırmızıya dönmektedir ve yaya geçidinde karşıya geçmekte olan bir yaya bulunmaktadır."},
                ],
                "prompt_token_uzunlugu": 10,
                "yanit_token_uzunlugu": 24,
            },
        ]
