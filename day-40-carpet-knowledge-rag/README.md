# Day 40: Tekstil ve Üretim Teknik Dokümanları Üzerinde Sektörel RAG Sistemi (Carpet Knowledge RAG)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg?style=flat-square&logo=scipy)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; halı ve tekstil fabrikalarında çalışan üretim mühendisleri, kalite kontrol laboratuvarları ve dokuma operatörleri için **ISO/TSE standartları, jakarlı tezgah ayar reçeteleri ve terbiye/apre kimyası dokümanları** üzerinde çalışan, halüsinasyonu sıfırlayan ve kaynak alıntılı (citation-grounded) yanıtlar üreten **Sektörel RAG (Retrieval-Augmented Generation) Motorudur**.

---

## 📖 Mentorluk Dersi ve 4 Modüler Faz Derinliği

```
                        ┌──────────────────────────────────────────────────────────┐
                        │      MÜHENDİS / OPERATÖR TEKNİK SORUSU (PROMPT)          │
                        │   "Akrilik halılarda fiksaj sıcaklığı kaç derece?"       │
                        └────────────────────────────┬─────────────────────────────┘
                                                     │
                                                     ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  40.1: SEMANTİK BAŞLIK DUYARLI PARÇALAMA (Header-Aware Chunking & Metadata Enrichment)      │
    │  - TS EN ISO 2060, ISO 105-X12, Stenter Ram Apre Dokümanları                                 │
    │  - Alt Başlık, Kategori ve Standart Referansını Koruyan Sliding Window (L=350, O=50)        │
    └────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                             │
                                             ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  40.2: METADATA FİLTRELİ HİBRİT VEKTÖR DEPOSU (Dense + TF-IDF Sparse Search)                 │
    │  - S_toplam = 0.60 * S_dense + 0.40 * S_sparse                                               │
    │  - Metadata Filtresi: [kategori = 'apre_kimyasal', standart = 'ISO']                        │
    └────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                             │
                                             ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  40.3: RERANKING, CONTEXT INJECTION & HALÜSİNASYON GÜVENLİK BARİYERİ (Guardrail)             │
    │  - Güven Eşiği Denetimi: Skor < 0.20 -> REDDEDİLDİ (Bilgi Tabanında Yok)                     │
    │  - Doğrulanmış Sektörel Prompt Enjeksiyonu + Alıntı Haritası ([Standart: TS EN ISO 2060])   │
    └────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                             │
                                             ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  40.4: DOĞRULANMIŞ TEKNİK YANIT & 6 PANELLİ TEŞHİS PANOSU                                    │
    │  "Lateks sırt kaplama ve aprede kurutma sıcaklığı 145°C - 155°C olmalıdır..."                │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 40.1: Sektörel Doküman Korpusu ve Başlık Duyarlı Parçalama
* **Standart Korpusu:** TS EN ISO 2060 (İplik Numara/Büküm), Vandewiele Uyumlu Jakar Dokuma Kılavuzu (Çözgü tansiyonu $45 \pm 5\text{ cN}$), ISO 105-X12 (Sürtünme Haslığı), Stenter Apre Talimatı ($145-155^\circ\text{C}$).
* **Başlık Duyarlılığı (Header-Aware):** Parçalama esnasında metnin ait olduğu alt başlık (`1.1`, `2.1` vb.) ve standart referansı metadata olarak her chunk'a iliştirilir.

### 40.2: Metadata Filtreli Hibrit Vektör Deposu
Teknik terimler (Örn: *Alpha_m, Nm 28/2, cN/tex, florokarbon, C6-emülsiyon*) kesin sözcük eşleşmesi (Sparse TF-IDF) gerektirirken, kavramsal sorular (*"dokumada gerginlik sınırları"*) anlamsal yoğun vektör (Dense Embedding) gerektirir.

$$S_{\text{hibrit}} = 0.60 \cdot S_{\text{dense}}(\mathbf{q}, \mathbf{d}) + 0.40 \cdot S_{\text{tfidf}}(\mathbf{q}, \mathbf{d})$$

### 40.3: Context Injection ve Halüsinasyon Güvenlik Bariyeri
* **Prompt Enjeksiyonu:** LLM'e sadece getirilen doğrulanmış standart metni verilir; dışarıdan tahmin yapması yasaklanır.
* **Reddetme Eşiği ($\theta = 0.20$):** Bilgi tabanında yer almayan alan dışı sorular (Örn: *"Mars roket yakıtı nedir?"*) doğrudan tespit edilerek reddedilir.

---

### 40.4: Soru-Cevap Deney Sonuçları

| Soru | Getirilen Standart | Doküman ID | Hibrit Skor (%) | Durum |
|---|---|---|---|---|
| **"Akrilik halı fiksaj ve kurutma sıcaklığı kaç derece olmalıdır?"** | Tekstil Kimyası ve Terbiye Talimatı | `DOC-FINISH-04` | **%78.42** | `BASARILI_YANIT` |
| **"Jakarlı tezgahlarda çözgü gerginlik limitleri nedir?"** | Vandewiele Dokuma Kılavuzu | `DOC-WEAVE-02` | **%82.15** | `BASARILI_YANIT` |
| **"Uzay mekiklerinde titanyum alaşım oranı nedir?"** | — (Alan Dışı) | — | **%0.00** | `REDDEDILDI_BILGI_YOK` |

---

## 🛠️ Dizin Yapısı

```
day-40-carpet-knowledge-rag/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # numpy, scipy, matplotlib, seaborn, pytest
├── ana_akis.py                      # Uçtan uca sektörel RAG soru-cevap ve analiz betiği
├── README.md                        # 220+ satır sektörel ve matematiksel dokümantasyon
├── src/
│   ├── __init__.py
│   ├── sektor_korpusu.py            # TS EN ISO, Jakar ve Terbiye teknik doküman korpusu
│   ├── semantik_parcalayici.py      # 40.1: Başlık duyarlı & overlap metin parçalayıcı
│   ├── vektor_deposu.py             # 40.2: Metadata filtreli hibrit vektör arama motoru
│   ├── rag_asistani.py              # 40.3: Reranking, prompt context injection & guardrail
│   └── gorsellestirici.py           # 40.4: 6 panelli sektörel RAG teşhis panosu
├── testler/
│   ├── __init__.py
│   └── test_carpet_rag.py           # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── sektorel_rag_paneli.png      # 6 panelli yüksek çözünürlüklü teşhis görseli
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

#

---

### 🔍 Dondurulmuş Mimari Analizleri (Freezing Architecture Rationale)

### 1. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- Halı yıkama, leke çıkarma, iplik türleri ve bakım talimatları içeren uzman teknik bilgi tabanını RAG mimarisiyle akıllı danışmana dönüştürmek için.

### 2. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- Müşteri temsilcilerinin teknik el kitaplarında saatlerce arama yapma ihtiyacını ortadan kaldırır; doğru reçeteyi saniyeler içinde verir.

### 3. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar, Limitler ve Dikkat Edilmesi Gerekenler)
- Teknik dokümanlarda bulunmayan veya çelişkili bilgiler içeren sorulara doğrudan yanıt üretemez.

### 4. 🔄 Alternatif Sistemler & Karşılaştırmalı Yaklaşımlar
- Geleneksel arama motoru, Chatbot karar ağaçları veya Fine-Tuned Domain LLM.

---

## 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Alana Özel RAG** | *Domain-Specific Knowledge RAG* | Halı dokuma teknikleri, iplik türleri, yöresel desenler ve bakım talimatlarından oluşan kurumsal bilgi tabanı RAG sistemi. |
| **Anlamsal Parçalama** | *Semantic Document Chunking* | Teknik dokümanların paragraf ve başlık bütünlüğünü koruyarak vektörleştirilmesi. |
| **Metaveri Filtreleme** | *Metadata-Filtered Retrieval* | Aramayı yalnızca belirli bir yöre veya iplik kategorisine sınırlayarak doğruluk artıran filtreleme. |
| **Kaynak Gösterme (Citations)** | *Grounded Source Attribution* | Üretilen cevabın hangi katalog ve teknik kılavuz sayfasından alındığını belirten şeffaf alıntılama. |

---

## 2. Ana Akışın Çalıştırılması
```bash
python ana_akis.py
```

### 3. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Birden fazla dokümandan gelen bilgileri tek bir teknik raporda birleştiren **"Çoklu Doküman Karşılaştırmalı Sentez Motoru (Cross-Document Synthesizer)"** fonksiyonu geliştirmek.

**Tamamlanan Çözüm:**
```python
def coklu_standart_karsilastir(self, sorular: list) -> list:
    karsilastirma_raporu = []
    for soru in sorular:
        sonuc = self.yanit_uret(soru, top_k=1)
        if sonuc["durum"] == "BASARILI_YANIT":
            kaynak = sonuc["kaynaklar"][0]
            karsilastirma_raporu.append({
                "parametre": soru,
                "standart": kaynak["kaynak_standart"],
                "alt_baslik": kaynak["alt_baslik"],
                "guven_skoru": f"%{kaynak['skor']*100:.1f}"
            })
    return karsilastirma_raporu
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Tekstil ve endüstriyel üretim hatlarında genel amaçlı açık uçlu bir LLM yerine neden **Doğrulanmış Sektörel RAG ve Güven Eşiği Bariyeri (Confidence Threshold Guardrail)** kullanılmalıdır?

> **Mentor Cevabı:**
> Endüstriyel tekstil üretiminde apre fiksaj sıcaklığının $150^\circ\text{C}$ yerine $170^\circ\text{C}$ olarak yanlış söylenmesi (LLM halüsinasyonu), on binlerce metrekarelik akrilik halının sararmasına ve liflerinin erimesine yol açar.
> 1. **Doğrulanmış Alıntı (Citation Grounding):** Sektörel RAG, cevabın hangi ISO standardından (`TS EN ISO 2060`) veya fabrika kılavuzundan geldiğini açıkça bağlar.
> 2. **Halüsinasyon Reddetme:** Model parametreyi bilmediğinde uydurmak yerine `REDDEDILDI_BILGI_YOK` kararı vererek operatörü fabrika laboratuvarına yönlendirir ve hatalı üretimi engeller.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
