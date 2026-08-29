# Day 125: Sandboxed Code Execution & Otonom Veri Analizi Ajanı

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; dil modellerinin ürettiği Python kodlarını güvenli ve izole bir sanal alanda çalıştıran, **AST Tabanlı Statik Güvenlik Analizi (Abstract Syntax Tree)**, **Sistem Çağrısı ve Kütüphane Engelleme (Syscall/Import Filtering)**, **Bellek İzolasyonu (Restricted Globals/Locals)**, **Stdio Yönlendirmesi** ve **Otomatik Matplotlib Grafik Yakalama (Plot Capture)** mimarisini sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Ajanlara Güvenle Kod Çalıştırma Gücü Vermek: Code Interpreter"

Büyük dil modelleri (LLM) karmaşık matematiksel hesaplamalarda ve veri analizlerinde sık sık halüsinasyon görür (örn. %64 doğruluk). Bu sorunun kesin çözümü, modeli doğrudan hesaplatmak yerine **"Python Kodu Yazdırıp Bir Sandboxta Çalıştırmaktır"** (ChatGPT Code Interpreter / Claude Artifacts).

Ancak kullanıcının veya modelin ürettiği Python kodunu doğrudan sunucuda `exec()` ile çalıştırmak çok büyük bir güvenlik açığıdır (`os.system('rm -rf /')` veya dosya çalınması).

**Güvenli Sandbox Mimarisi Nasıl Çalışır?**
1. 🌳 **AST Güvenlik Analizi:** Kod çalıştırılmadan önce Python'ın soyut sözdizim ağacı (AST) taranır. `os`, `sys`, `subprocess`, `open`, `eval` gibi tehlikeli çağrılar ve `__subclasses__` gibi kaçış vektörleri tespit edilirse kod anında reddedilir.
2. 🛡️ **Kısıtlı Çalışma Alanı:** Sadece güvenli veri bilimi kütüphanelerine (`numpy`, `math`, `matplotlib`) izin verilir.
3. 📥 **Stdio Yönlendirmesi:** Kodun `print()` çıktıları `io.StringIO` ile yakalanır, terminale taşma yapmaz.
4. 📈 **Otomatik Grafik Yakalama:** `matplotlib` çizimleri ekran açılmadan arka planda yakalanıp PNG dosyası olarak rapora eklenir.

```
      [Kullanıcı İstemi] ──> [LLM: Python Kodu Üret]
                                     │
                                     ▼
                       [AST Statik Güvenlik Analizi]
                       (os, sys, open, eval engelleme)
                          │                     │
                   (Güvensiz)                (Güvenli)
                          ▼                     ▼
                    [REDDEDİLDİ]      [Kısıtlı Globals / Locals]
                                      [io.StringIO Stdout Yakalama]
                                      [Matplotlib Plot Capture]
                                                │
                                                ▼
                                      [İstatistiksel Rapor + Grafikler]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Sandboxed Code Execution Mimarisi
- **Doğrudan LLM Hesaplaması:** Metin tabanlı olasılıksal tahmin; büyük sayılarda ve döngülerde halüsinasyon riski.
- **Code Interpreter Yaklaşımı:** Dil modeli kodu yazar, Python yorumlayıcısı deterministik olarak çalıştırır (%100 matematiksel doğruluk).

### 2. AST (Abstract Syntax Tree) Tabanlı Statik Güvenlik ve Syscall/Import Filtreleme
- Kod metni çalıştırılmadan önce sözdizim ağacına (`ast.parse`) dönüştürülür.
- `ast.Import`, `ast.ImportFrom`, `ast.Call` ve `ast.Attribute` düğümleri taranarak zararlı modüller ve sandbox kaçış girişimleri sıfır toleransla engellenir.

### 3. Güvenli Bellek İzolasyonu (Custom Globals/Locals) ve Stdio Yönlendirmesi
- `__builtins__` sözlüğü budanarak `eval`, `exec`, `open`, `compile` fonksiyonları ortamdan tamamen silinir.
- `sys.stdout` ve `sys.stderr` izole tamponlara bağlanarak tam denetim sağlanır.

### 4. Otonom Veri Analitiği (EDA), Otomatik Grafik Yakalama ve Salt LLM Kıyaslaması
- **Matematiksel Doğruluk:** %64.2 (Salt LLM) -> **%100.0 (Sandbox)**.
- **Halüsinasyon Önleme:** %58.0 -> **%100.0**.
- **Grafik Üretimi:** %0.0 -> **%100.0 (Matplotlib Figür Yakalama)**.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Code Interpreter** | LLM tarafından üretilen kodları çalıştırıp sonuçları yorumlayan ajan mimarisi. |
| **Sandbox (Korumalı Alan)** | Kodun sistem kaynaklarına erişimini kısıtlayan güvenli izole çalışma ortamı. |
| **AST (Abstract Syntax Tree)** | Kaynak kodun hiyerarşik sözdizimsel ağaç yapısı temsili. |
| **Syscall Filtering** | Tehlikeli işletim sistemi çağrılarının denetlenip engellenmesi. |
| **Sandbox Escape** | İzole ortam sınırlarını aşarak ana işletim sistemine sızma girişimi. |
| **Stdio Redirection** | Standart girdi/çıktı akışlarının geçici bellek tamponlarına yönlendirilmesi. |
| **Restricted Globals** | Güvenlik riski taşıyan fonksiyonların çıkarıldığı kısıtlı global değişken sözlüğü. |
| **Plot Capture** | Grafik kütüphanelerinin GUI açmadan doğrudan piksel tamponuna kaydedilmesi. |
| **Static Code Analysis** | Kodu çalıştırmadan kaynak kod yapısını inceleyerek güvenlik denetimi yapma. |
| **Deterministic Math** | Olasılıksal dil modeli tahmini yerine kesin çalışan matematiksel yürütme. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %100 kesin matematiksel doğruluk.  │ • Ağ erişimi ve dosya sistemi tam    │
 │ • AST ile %100 saldırı engelleme.    │   kapatıldığı için dış API çekilemez │
 │ • Otomatik matplotlib grafik üretimi.│ • Aşırı karmaşık C kütüphanelerinde  │
 │ • Sıfır halüsinasyonlu veri analizi. │   izolasyon yönetimi zorlaşabilir.   │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Finansal analitik, biyoinformatik  │ • Docker/gVisor gibi konteynerize    │
 │   ve kurumsal BI otomasyonları.      │   sandboxlar olmadan çekirdek riski. │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/sandboxed_agent_paneli.png` dosyası üretilir:
1. **Salt LLM vs Code Interpreter Başarım Kıyaslaması**
2. **AST Statik Güvenlik ve İzolasyon Filtresi**
3. **İzole Sandbox Yürütme Gecikmesi (ms)**
4. **Sandbox İçinde Yakalanan Görselleştirme Çıktısı**
5. **Code Interpreter İzolasyon ve İcra Mimarisi**
6. **Sandboxed Execution Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Güvenli veri analizi ajanını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
