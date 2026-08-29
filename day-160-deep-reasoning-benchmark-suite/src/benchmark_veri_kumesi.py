"""
Derin Akıl Yürütme Benchmark Veri Kümesi Modülü (Day 160 - FAZ 8 BÜYÜK FİNALİ).
AIME, GPQA Diamond ve ARC-Challenge zorlu muhakeme problemlerini ve altın cevaplarını barındırır.
"""

from typing import Dict, Any, List


class BenchmarkVeriKumesi:
    """AIME, GPQA ve ARC-Challenge test paketini sağlayan veri modülü."""

    PROBLEMLER = [
        # AIME (Olimpiyat Matematiği)
        {
            "id": "aime_01",
            "benchmark": "AIME",
            "soru": "Tüm k pozitif tam sayıları için 2^k + 3^k toplamının bir tam kare olmasını sağlayan k değerlerini bulunuz.",
            "kategori": "Sayılar Teorisi",
            "altin_cevap": "1",
            "zorluk_seviyesi": "Doktora / Olimpiyat",
        },
        {
            "id": "aime_02",
            "benchmark": "AIME",
            "soru": "Bir polinom P(x) için P(P(x)) = (P(x))^2 + x denklemi veriliyor. P(0)'ın alabileceği değerler toplamı kaçtır?",
            "kategori": "Cebir",
            "altin_cevap": "0",
            "zorluk_seviyesi": "Olimpiyat",
        },
        # GPQA Diamond (Doktora Seviyesi Fen Bilimleri)
        {
            "id": "gpqa_01",
            "benchmark": "GPQA Diamond",
            "soru": "Kuantum elektrodinamiğinde (QED) 1-loop elektron öz-enerji diyagramının infrared divergent terimi hangi Ward kimliği ile iptal edilir?",
            "kategori": "Teorik Fizik",
            "altin_cevap": "Ward-Takahashi",
            "zorluk_seviyesi": "Doktora / Uzman",
        },
        {
            "id": "gpqa_02",
            "benchmark": "GPQA Diamond",
            "soru": "DNA replikasyonunda Okazaki parçacıklarının birleştirilmesi sırasında RNA primerini kesip uzaklaştıran ve niki onaran ana enzim çifti hangisidir?",
            "kategori": "Moleküler Biyoloji",
            "altin_cevap": "RNase H ve DNA Ligaz",
            "zorluk_seviyesi": "Yüksek Lisans",
        },
        # ARC-Challenge (Soyut Muhakeme ve Fiziksel Çıkarım)
        {
            "id": "arc_01",
            "benchmark": "ARC-Challenge",
            "soru": "Kapalı bir cam fanusta yanan mum ve yeşil bitki dengede durmaktadır. Işık kesilirse fanustaki O2/CO2 oranı nasıl değişir?",
            "kategori": "Fiziksel / Biyolojik Mantık",
            "altin_cevap": "O2 azalır, CO2 artar",
            "zorluk_seviyesi": "İleri Muhakeme",
        },
        {
            "id": "arc_02",
            "benchmark": "ARC-Challenge",
            "soru": "Sürtünmesiz eğik düzlemde yuvarlanan içi boş silindir ile içi dolu silindirden hangisi tabana daha önce ulaşır?",
            "kategori": "Klasik Mekanik",
            "altin_cevap": "İçi dolu silindir",
            "zorluk_seviyesi": "Üniversite Fizik",
        },
    ]

    @classmethod
    def problemleri_getir(cls, benchmark_adi: str = None) -> List[Dict[str, Any]]:
        """İstenen benchmark'a ait soruları döner."""
        if benchmark_adi:
            return [p for p in cls.PROBLEMLER if p["benchmark"].lower() == benchmark_adi.lower()]
        return cls.PROBLEMLER
