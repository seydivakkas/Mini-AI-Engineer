# Day 126: Çok Katmanlı Ajan Bellek Sistemleri (Working, Episodic & Semantic Vector Memory)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; dil modeli ajanlarının oturumlar ve haftalar boyunca kullanıcı tercihlerini, olguları ve olay geçmişini hatırlamasını sağlayan **Çok Katmanlı Bellek Hiyerarşisi (Working, Episodic, Semantic, Procedural)**, **Mem0 / Zep Tarzı Olgu Çıkarımı (Fact Extraction)**, **Çelişki Giderme (ADD / UPDATE / NOOP)** ve **Ebbinghaus Unutma Eğrili Hibrit Hatırlama Motoru** mimarisini sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Ajanlara Kalıcı Hafıza Kazandırmak: Mem0 & Çok Katmanlı Bellek"

Standart dil modelleri her yeni oturumda sıfırdan başlar (**Stateless**). Sohbet geçmişini modele vermenin en ilkel yolu son 5 mesajı prompta eklemektir (**Kayan Pencere / Sliding Window**). Ancak bu yaklaşım:
1. 10 tur öncesindeki kritik kullanıcı tercihini unutur (%24 hatırlama oranı),
2. Kullanıcı fikrini değiştirdiğinde ("Artık PyTorch değil, JAX kullanıyorum") eski ve yeni bilgi arasında çelişkiye düşer,
3. Tüm sohbeti prompta tıkıştırarak devasa token israfına yol açar.

**Modern Çok Katmanlı Bellek Mimarisi (Mem0 / Zep):**
- ⚡ **1. Çalışma Belleği (Working Memory):** Anlık konuşmanın son birkaç turunu tutan hızlı RAM.
- 📜 **2. Episodik Bellek (Episodic Memory):** Geçmiş görevlerin ne zaman, nasıl ve hangi sonuçla yapıldığını kaydeden günlük.
- 🧠 **3. Semantik Vektör Belleği (Semantic Memory):** Kullanıcının tercihlerini, uzmanlık alanlarını ve kısıtlarını anlamsal vektör uzayında saklayan kalıcı disk.
- 🔄 **4. Çelişki Giderme:** Yeni bir olgu geldiğinde eski kayıtla benzerliği kıyaslanır. Çelişki varsa eski kayıt geçersiz kılınır (`UPDATE`), yeniyse eklenir (`ADD`), aynıysa es geçilir (`NOOP`).
- ⏳ **5. Ebbinghaus Unutma Eğrisi:** Uzun süre kullanılmayan anıların skoru zamansal olarak azalır ($R = e^{-\lambda \cdot t}$).

```
  [Kullanıcı Mesajı] ──> [Olgu & Tercih Çıkarımı]
                                │
                                ▼
                  [Semantik Vektör Karşılaştırma]
                     ┌──────────┬──────────┐
                     │          │          │
                (Sim > 0.90) (Çelişki) (Yeni Tercih)
                     ▼          ▼          ▼
                  [NOOP]     [UPDATE]    [ADD]
                 (Aynı Olgu) (Eskiyi Sil)(Yeni Kayıt)
                                │
                                ▼
                  [Hibrit Arama (Sim + Recency + Imp)]
                                │
                                ▼
                  [Kişiselleştirilmiş Prompt Enjeksiyonu]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Çok Katmanlı Bellek Mimarisi
- **Working Memory:** Bağlam penceresinin en güncel $K$ turunu tutar.
- **Episodic Memory:** Zaman damgalı görev geçmişini saklar.
- **Semantic Memory:** Vektör gömmeleri ile indekslenmiş kullanıcı profilini tutar.
- **Procedural Memory:** Araç çağırma ve ajan iş akışı kurallarını saklar.

### 2. Mem0 / Zep Paradigması: Dinamik Olgu Çıkarma, Çelişki Giderme ve Bellek Güncelleme
- Model konuşma akışından kritik olguları ayıklar.
- Yeni tercih ("JAX kullanıyorum") eski tercih ("PyTorch kullanıyorum") ile karşılaştırılarak eski kayıt arşive alınır ve güncel bilgi hafızaya yazılır.

### 3. Ebbinghaus Unutma Eğrisi, Zamansal Bozunma (Temporal Decay) ve Hibrit Hatırlama Puanı
- Bellek erişim formülü:
  $$S(m) = w_{\text{sim}} \cdot \text{CosineSim}(q, m) + w_{\text{rec}} \cdot e^{-\lambda \cdot \Delta t} + w_{\text{imp}} \cdot \text{Importance}(m)$$
- Sık erişilen anılar pekiştirilir; alakasız anılar arka plana itilir.

### 4. Uzun Süreli Kişiselleştirme ve Kayan Pencereye Karşı %96.5 Hatırlama Başarımı
- **10+ Tur Sonra Hatırlama:** %24.0 (Kayan Pencere) -> **%96.5 (Çok Katmanlı Bellek)**.
- **Tercih Çelişkisi Giderme:** %18.5 -> **%94.0**.
- **Token Tasarrufu:** Devasa sohbet logları yerine yalnızca en ilgili 2-3 anı enjekte edilerek **%92.5 verimlilik** sağlanır.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Working Memory** | Anlık konuşma bağlamını tutan kısa süreli kayan pencere hafızası. |
| **Episodic Memory** | Görevlerin, eylemlerin ve olayların kronolojik deneyim kaydı. |
| **Semantic Memory** | Vektör uzayında saklanan kullanıcı tercihleri ve olgusal kalıcı bellek. |
| **Procedural Memory** | Ajanın araç kullanma ve görev icra etme kurallarını saklayan işlem hafızası. |
| **Fact Extraction** | Serbest metin içinden yapılandırılmış kullanıcı olgularını ayıklama süreci. |
| **Conflict Resolution** | Yeni bilgi ile çelişen eski hafıza kayıtlarının güncellenmesi veya silinmesi. |
| **Ebbinghaus Forgetting Curve** | Kullanılmayan anıların hatırlanma olasılığının zamanla üstel azalması ($e^{-\lambda t}$). |
| **Recency Decay** | Bir hafıza kaydına en son ne zaman erişildiğine bağlı tazelik puanı. |
| **Hybrid Retrieval** | Anlamsal benzerlik, tazelik ve önemi ağırlıklandırarak en uygun anıyı seçme. |
| **Mem0 Architecture** | Ajanlara dinamik ve kişiselleştirilmiş uzun süreli hafıza kazandıran çerçeve. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • 10+ oturum boyunca %96.5 hatırlama.│ • Her turda olgu çıkarma ve vektör   │
 │ • Otomatik çelişki giderme (UPDATE). │   arama nedeniyle ek gecikme (latency)│
 │ • %92.5 token tasarrufu.             │ • Yanlış çıkarılan olguların hafızayı│
 │ • Ebbinghaus tabanlı akıllı unutma.  │   kirletme (hallucinated memory) riski│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kişiselleştirilmiş AI asistanları, │ • KV-Cache ve veritabanı senkronizas-│
 │   akıllı CRM botları ve otonom ajanlar│  yonunda dağıtık ölçekleme zorluğu.  │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/agent_memory_paneli.png` dosyası üretilir:
1. **Kayan Pencere vs Çok Katmanlı Bellek Başarımı**
2. **Ebbinghaus Unutma & Tazelik (Recency Decay) Eğrisi**
3. **Hibrit Hatırlama Puan Formülü Bileşenleri**
4. **Katman Başına Aktif Bellek Atomu Dağılımı**
5. **Mem0 / Zep Bellek Yaşam Döngüsü ve Çelişki Giderme**
6. **Ajan Bellek Mimarisi Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Çok katmanlı hafızalı ajanı çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
