"""Kirli ve Düzensiz Sentetik Tabüler Görsel Metaveri Üreticisi.

Bu modül; eksik veriler (NaN/None), aykırı uç değerler (outliers), veri tipi
tutarsızlıkları ve duplikasyonlar içeren gerçekçi bir endüstriyel görsel veri seti üretir.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def kirli_veri_kumesi_uret(
    satir_sayisi: int = 500,
    rastgele_tohum: int = 42
) -> pd.DataFrame:
    """Temizleme boru hatlarını test etmek için kusurlu ve kirli bir DataFrame üretir.

    İçerdiği Hatalar:
        1. Sütunlarda rastgele %5-%10 oranında eksik veri (NaN).
        2. Genişlik ve parlaklık değerlerinde aşırı uç aykırı değerler (outliers).
        3. Tamamen aynı olan mükerrer (duplicate) satırlar.
        4. Sayısal olması gerekirken string içine gömülmüş kirli değerler.
        5. Fazla bellek tüketen 64-bit tipler ve 'object' kategorikler.

    Parametreler:
        satir_sayisi (int): Üretilecek ham satır adedi.
        rastgele_tohum (int): Tekrarlanabilirlik tohumu.

    Döndürür:
        pd.DataFrame: Kirli tabüler veri çerçevesi.
    """
    np.random.seed(rastgele_tohum)

    # 1. Temel sütunların üretimi
    kimlikler = [f"GORSEL_{i:05d}" for i in range(satir_sayisi)]
    genislikler = np.random.normal(loc=1920, scale=120, size=satir_sayisi)
    yukseklikler = np.random.normal(loc=1080, scale=80, size=satir_sayisi)
    dosya_boyutlari = np.random.exponential(scale=1500, size=satir_sayisi) + 200
    parlakliklar = np.random.normal(loc=128, scale=35, size=satir_sayisi)

    kategoriler = np.random.choice(
        ["Hereke_Ipek", "Usak_Klasik", "Kayseri_Yun", "Isparta_Modern"],
        size=satir_sayisi
    )
    kusur_durumlari = np.random.choice([0, 1], p=[0.85, 0.15], size=satir_sayisi)

    veri = pd.DataFrame({
        "gorsel_kimligi": kimlikler,
        "genislik": genislikler,
        "yukseklik": yukseklikler,
        "dosya_boyutu_kb": dosya_boyutlari,
        "ortalama_parlaklik": parlakliklar,
        "kumas_turu": kategoriler,
        "kusurlu_mu": kusur_durumlari,
    })

    # 2. Yapay Aykırı Değerler (Outliers) Enjeksiyonu
    aykiri_indeksler = np.random.choice(satir_sayisi, size=int(satir_sayisi * 0.03), replace=False)
    veri.loc[aykiri_indeksler[:5], "genislik"] = 99999.0   # Aşırı yüksek saçma çözünürlük
    veri.loc[aykiri_indeksler[5:10], "genislik"] = -500.0  # Negatif çözünürlük hatası
    veri.loc[aykiri_indeksler[10:], "ortalama_parlaklik"] = 1500.0  # 255 üstü taşmış parlaklık

    # 3. Eksik Veri (NaN) Enjeksiyonu
    eksik_indeks_genislik = np.random.choice(satir_sayisi, size=int(satir_sayisi * 0.06), replace=False)
    eksik_indeks_kategori = np.random.choice(satir_sayisi, size=int(satir_sayisi * 0.05), replace=False)
    eksik_indeks_boyut = np.random.choice(satir_sayisi, size=int(satir_sayisi * 0.08), replace=False)

    veri.loc[eksik_indeks_genislik, "genislik"] = np.nan
    veri.loc[eksik_indeks_kategori, "kumas_turu"] = np.nan
    veri.loc[eksik_indeks_boyut, "dosya_boyutu_kb"] = np.nan

    # 4. Mükerrer (Duplicate) Satırlar Eklenmesi
    yinelemeler = veri.sample(n=int(satir_sayisi * 0.05), random_state=rastgele_tohum)
    veri = pd.concat([veri, yinelemeler], ignore_index=True)

    return veri
