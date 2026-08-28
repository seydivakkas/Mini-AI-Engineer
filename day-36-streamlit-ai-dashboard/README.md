# Day 36: Streamlit ile İnteraktif AI Kontrol Paneli (Interactive AI Dashboard)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Pillow](https://img.shields.io/badge/Pillow-9.5+-005571.svg?style=flat-square)](https://python-pillow.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; doğal dil işleme, bilgisayarla görme ve doküman tabanlı RAG asistanı yeteneklerini tek bir görsel kullanıcı arayüzünde (GUI) birleştiren, gerçek zamanlı çıkarım ve model telemetrisi sunan **Streamlit Çoklu Görev AI Kontrol Paneli** mimarisidir.

---

## 📖 Mentorluk Dersi ve Teorik Derinlik

### 1. Streamlit Yürütme Modeli (Re-execution Paradigm)
Streamlit, klasik web framework'lerinden (React, Vue) farklı olarak deklaratif ve betik tabanlı bir yürütme modeli kullanır:
- **Tetikleyici Yeniden Çalıştırma (Rerun):** Kullanıcı arayüzdeki herhangi bir bileşene (slider, buton, metin kutusu) dokunduğunda, Python dosyasının tamamı baştan sona tekrar çalıştırılır.
- **Durum Yönetimi (`st.session_state`):** Yeniden çalıştırmalar sırasında model ağırlıklarının, yüklenen dosyaların ve sohbet geçmişinin silinmesini önlemek için oturum hafızası (`session_state`) kullanılır.
- **Model Önbellekleme (`@st.cache_resource`):** Ağır sinir ağı modellerinin her tıklamada baştan RAM'e yüklenmesini engelleyerek tekil (singleton) nesne olarak saklar.

```
                    ┌──────────────────────────────────────────────────────────┐
                    │               KULLANICI ETKİLEŞİMİ (UI EVENT)            │
                    │               (Slider, Buton, Dosya Yükleme)             │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │  STREAMLIT OTURUM & YÜRÜTME DÖNGÜSÜ (RE-EXECUTION CYCLE)                     │
        │  - st.session_state Kontrolü (Sohbet Geçmişi, Sonuçlar)                      │
        │  - @st.cache_resource ile Önbellekli AI Motoru Çağrısı                      │
        │  - Sidebar Parametreleri (Threshold, Model Seçimi)                          │
        └────────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
        ┌───────────────────────────────┐                 ┌───────────────────────────────┐
        │  SEKME 1 & 2: MODALİTE MOTORU │                 │  SEKME 3 & 4: RAG VE TELEMETRİ│
        │  - Metin Sınıflandırma        │                 │  - st.chat_message Sohbet     │
        │  - Bounding Box Çizimi        │                 │  - Kaynak Atıf Rozetleri      │
        │  - Baskın RGB Renk Paleti     │                 │  - Canlı Gecikme Zaman Serisi │
        └───────────────────────────────┘                 └───────────────────────────────┘
                                                 │
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  MODERN, İNTERAKTİF VE PROFESYONEL AI KONTROL PANELİ     │
                    └──────────────────────────────────────────────────────────┘
```

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Streamlit Reaktif Yürütme** | *Streamlit Reactive Execution* | Kullanıcı her etkileşimde (buton, slider) tüm betiği yukarıdan aşağıya yeniden koşturan reaktif UI paradigması. |
| **Oturum Durumu (Session State)** | *Session State Management (`st.session_state`)* | Yeniden çalıştırmalar arasında değişkenlerin, model ağırlıklarının ve kullanıcı verilerinin bellekte korunması. |
| **Önbellekleme (`@st.cache_resource`)** | *Resource Caching* | Ağır yapay zeka modellerinin belleğe yalnızca bir kez yüklenmesini ve her istekte tekrar yüklenmesini önleyen dekoratör. |
| **İnteraktif Görselleştirme** | *Real-Time Inference Dashboard* | Kullanıcının yüklediği görsel üzerinde model çıkarımını ve metrik grafiklerini anında sunan web arayüzü. |

---

## 2. Çoklu Sekmeli (Multi-Tab) Kontrol Paneli Mimarisi

Panomuz 4 ana işlevsel sekmeye ayrılmıştır:
1. **📝 Metin Analizi & Sınıflandırma:** Kullanıcı metinlerini 3 ana kategoriye sınıflandırır, güven dağılımı ilerleme çubuklarını gösterir ve 64-boyutlu embedding vektörünü sunar.
2. **🖼️ Bilgisayarlı Görü & Kusur Tespiti:** Yüklenen görseller üzerinde Pillow ile dinamik sınırlayıcı kutular (bounding boxes) çizer, anomali koordinatlarını tablo olarak listeler ve baskın RGB renk analizini yapar.
3. **💬 RAG Doküman Asistanı:** `st.chat_input` ve `st.chat_message` bileşenleriyle doküman tabanlı sohbet sağlar; her yanıta doğrulanabilir kaynak rozetleri (`[Kaynak: KB-01]`) ekler.
4. **📈 Sistem Telemetrisi:** Yapılan toplam çıkarım sayısını ve model gecikme geçmişini çizgi grafik olarak gerçek zamanlı görselleştirir.

---

### 3. Dashboard Yetkinlik ve Özellik Tablosu

| Sekme / Modül | Girdi Türü | Görselleştirme Çıktısı | Model Türü |
|---|---|---|---|
| **Metin Analizi** | `st.text_area` | Güven Çubukları (`st.progress`), Vektör JSON | PyTorch Attention Classifier |
| **Görüntü Analizi** | `st.file_uploader` | Bounding Box Canvas, Renk Swatch | Pillow + Vision Defect Sim |
| **RAG Asistanı** | `st.chat_input` | Sohbet Balonları (`st.chat_message`) | Vector Semantic Retriever |
| **Telemetri** | Oturum Olayları | Canlı Çizgi Grafik (`st.line_chart`) | In-Memory Telemetry Logger |

---

## 🛠️ Dizin Yapısı

```
day-36-streamlit-ai-dashboard/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # streamlit, torch, pillow, matplotlib, seaborn, pytest
├── app.py                           # Streamlit web uygulamasının ana giriş noktası
├── ana_akis.py                      # Headless dashboard simülatörü ve görsel üretici
├── README.md                        # Detaylı teorik ve mimari dokümantasyon (220+ Satır)
├── src/
│   ├── __init__.py
│   ├── ai_modulleri.py              # Önbellekli Metin, Görüntü ve RAG motorları
│   ├── bilesenler.py                # Yeniden kullanılabilir Streamlit arayüz bileşenleri
│   ├── panel_yoneticisi.py          # Streamlit sekmeleri, session_state ve sayfa düzeni
│   └── gorsellestirici.py           # 6 panelli dashboard analiz panosu (Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_streamlit_dashboard.py  # 6 adet kapsamlı birim test
└── ciktilar/
    └── streamlit_dashboard_paneli.png # 6 panelli dashboard analiz görseli
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Streamlit Web Uygulamasının Başlatılması
```bash
streamlit run app.py
```
*Uygulama otomatik olarak tarayıcınızda `http://localhost:8501` adresinde açılacaktır.*

### 3. Ana Akış ve Teşhis Panosunun Çalıştırılması
```bash
python ana_akis.py
```

### 4. Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** `src/panel_yoneticisi.py` içerisine **"Karanlık / Aydınlık Tema Değiştirici (Theme State Switcher)"** ekleyerek kullanıcının seçimine göre arayüz grafiklerinin renk paletini (`dark_mode=True/False`) dinamik güncelleyen bir arayüz fonksiyonu yazmak.

**Tamamlanan Çözüm:**
```python
def tema_ayarla():
    if "karanlik_mod" not in st.session_state:
        st.session_state["karanlik_mod"] = True
    
    with st.sidebar:
        secim = st.toggle("🌙 Karanlık Mod", value=st.session_state["karanlik_mod"])
        if secim != st.session_state["karanlik_mod"]:
            st.session_state["karanlik_mod"] = secim
            st.rerun()
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Streamlit ile büyük bir derin öğrenme modelini (örneğin ResNet veya Stable Diffusion) yayına alırken neden model yükleme fonksiyonunun üzerine `@st.cache_resource` eklemek zorunludur?

> **Mentor Cevabı:**
> 1. **Bellek Sızıntısı ve Rerun Maliyeti:** Streamlit'te kullanıcı her butona bastığında veya slider kaydırdığında dosya baştan çalıştırılır. `@st.cache_resource` kullanılmazsa, her tıklamada gigabaytlarca ağırlık RAM'e/VRAM'e yeniden yüklenir, saniyelerce gecikme oluşur ve sunucu birkaç dakika içinde bellek taşması (Out Of Memory - OOM) ile çöker.
> 2. **Singleton Paylaşımı:** `@st.cache_resource`, model örneğini global hafızada tekil (singleton) olarak tutar ve oturumlar arasında güvenle paylaşarak sıfır yükleme maliyetiyle anında çıkarım yapılmasını sağlar.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
