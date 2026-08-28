"""
Day 34: Mini RAG Asistanı & Doküman Soru-Cevap Motoru Birim Testleri.
"""

import os
import pytest
from src.metin_parcalayici import MetinParcalayici
from src.vektor_deposu import VektorDeposu
from src.rag_ureteci import RAGUreteci
from src.rag_asistani import MiniRAGAsistani
from src.gorsellestirici import RAGGorsellestirici


def test_metin_parcalayici_boyut_ve_cakisma():
    """Metin parçalama boyutu ve çakışma testi."""
    parcalayici = MetinParcalayici(chunk_boyutu=10, cakisma_miktari=3)
    metin = " ".join([f"kelime_{i}" for i in range(25)])

    parcalar = parcalayici.parcala("DOC-1", "Test", metin)
    assert len(parcalar) >= 3
    assert parcalar[0]["chunk_id"] == "DOC-1_chunk_00"
    assert parcalar[0]["kelime_sayisi"] == 10
    # İkinci parçanın başlangıcı: 10 - 3 = 7
    assert parcalar[1]["baslangic_idx"] == 7


def test_metin_parcalayici_hata_firlatma():
    """Geçersiz çakışma parametresi hata testi."""
    with pytest.raises(ValueError):
        MetinParcalayici(chunk_boyutu=10, cakisma_miktari=10)


def test_vektor_deposu_indeksleme_ve_arama():
    """Vektör deposu ekleme ve kosinüs arama testi."""
    depo = VektorDeposu(embed_dim=64)
    parcalar = [
        {"chunk_id": "C1", "doc_id": "D1", "baslik": "Derin Aglar", "metin": "Evrisimli sinir aglari ve filtreler", "metaveri": {}},
        {"chunk_id": "C2", "doc_id": "D2", "baslik": "Veritabani", "metin": "SQL indeksleme ve B-Tree tablolari", "metaveri": {}}
    ]
    depo.parcalari_ekle(parcalar)

    sonuclar = depo.arama("evrisimli filtreler", top_k=1)
    assert len(sonuclar) == 1
    assert sonuclar[0]["chunk_id"] == "C1"


def test_rag_ureteci_prompt_ve_yanit():
    """Prompt hazırlama ve kaynak atıflı sentez testi."""
    uretec = RAGUreteci(guven_esigi=0.10)
    parcalar = [
        {"chunk_id": "C1", "doc_id": "D1", "baslik": "YOLO", "metin": "Kamera karelerinde nesne tespiti yapar", "skor": 0.85}
    ]
    prompt = uretec.prompt_hazirla("YOLO nedir?", parcalar)
    assert "[Kaynak: C1]" in prompt
    assert "YOLO nedir?" in prompt

    sentez = uretec.yanit_sentezle("YOLO nedir?", parcalar)
    assert sentez["durum"] == "BASARILI"
    assert "C1" in sentez["kaynaklar"]
    assert "[Kaynak: C1]" in sentez["yanit"]


def test_rag_ureteci_reddetme():
    """Yetersiz bilgi durumunda ret (refusal) testi."""
    uretec = RAGUreteci(guven_esigi=0.50)
    parcalar = [
        {"chunk_id": "C1", "doc_id": "D1", "baslik": "YOLO", "metin": "Nesne tespiti", "skor": 0.15}
    ]
    sentez = uretec.yanit_sentezle("Kuantum nedir?", parcalar)
    assert sentez["durum"] == "YETERSIZ_KANIT"
    assert len(sentez["kaynaklar"]) == 0


def test_mini_rag_asistani_entegrasyon():
    """Uçtan uca MiniRAGAsistani entegrasyon testi."""
    asistan = MiniRAGAsistani(chunk_boyutu=15, cakisma_miktari=5, embed_dim=64, guven_esigi=0.10)
    asistan.dokuman_ekle("D1", "RAG Mimarisi", "RAG mimarisi metin parcalama ve vektor indeksleme ile soru cevaplar.")

    cikis = asistan.soru_sor("RAG nasil calisir?", top_k=1)
    assert cikis["durum"] == "BASARILI"
    assert len(cikis["getirilen_parcalar"]) >= 1
    assert "D1_chunk_00" in cikis["kaynaklar"]


def test_rag_gorsellestirici(tmp_path):
    """6 panelli teşhis panosu görselleştirme testi."""
    mock_cikis = {
        "soru": "Test soru",
        "getirilen_parcalar": [
            {"chunk_id": "C1", "baslik": "Test", "skor": 0.45}
        ]
    }
    cikis_path = str(tmp_path / "test_rag.png")
    yol = RAGGorsellestirici.rag_paneli_ciz(mock_cikis, hedef_path=cikis_path)
    assert os.path.exists(yol)
