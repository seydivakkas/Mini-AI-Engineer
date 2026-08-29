# Day 119: LLM Knowledge Distillation ve Self-Instruct

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 6: İleri LLM Mimarileri, Hizalama (Alignment) ve RLHF / DPO / GRPO**  
> Bu modül; devasa Öğretmen Modellerin (Teacher LLM: 70B+) yüksek kaliteli akıl yürütme ve yumuşatılmış logit olasılık dağılımlarını hafif Öğrenci Modellere (Student LLM: 1B/3B) aktaran **Knowledge Distillation (KD), Self-Instruct Sentetik Akıl Yürütme ve Logit-Level KL Hizalama Motoru**nu sıfırdan inşa edip analiz eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Devlerden Cücelere Zekâ Aktarımı: Bilgi Damıtma"

70 milyar parametreli dev bir yapay zeka modelini (örneğin Llama-3-70B veya DeepSeek-V3) bir akıllı telefonda veya tek bir GPU'da çalıştırmak imkansızdır. Ancak 1 milyar parametreli küçük bir modeli sıfırdan eğitmeye kalktığımızda da zekâsı yetersiz kalır.

**Knowledge Distillation (Bilgi Damıtma)** tam olarak bir **Usta-Çırak İlişkisi** kurar:
1. 🧙‍♂️ **Öğretmen Model (Teacher):** Devasa ve çok bilgili bir modeldir. Sorulara verdiği yanıtlarda sadece doğru kelimeyi değil, diğer alternatif kelimelerin aralarındaki ince anlamsal ilişkileri de bilir (**Karanlık Bilgi / Dark Knowledge**).
2. 🧑‍🎓 **Öğrenci Model (Student):** 10 kat daha küçük ve 6.6 kat daha hızlı hafif bir modeldir.
3. 🌡️ **Sıcaklık Ölçekleme ($T$):** Logitler sıcaklıkla yumuşatılır; böylece öğretmen model çırağına *"Doğru cevap A ama B şıkkı da C'den çok daha mantıklı bir alternatif"* diyerek düşünce biçimini aktarır.
4. 📉 **KL Diverjansı:** Küçük öğrenci model, öğretmenin yumuşatılmış olasılık dağılımını ($D_{\text{KL}}$) taklit etmeyi öğrenir.
5. 🚀 **Sonuç:** 10 kat daha az bellek harcayan ve cep telefonunda saniyede yüzlerce kelime üretebilen süper zeki küçük bir model (örn. DeepSeek-R1-Distill-1.5B)!

```
       DEV ÖĞRETMEN MODEL (70B)                              KOMPAKT ÖĞRENCİ MODEL (1B)
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────────────┐
 │ • Girdi (x) -> logits_T              │            │ • Girdi (x) -> logits_S                      │
 │ • Softmax(logits_T / T) (Yumuşak)    │ ─────────> │ • Softmax(logits_S / T)                      │
 └──────────────────────────────────────┘     KL     └──────────────────────┬───────────────────────┘
                                           Diverjans                        │
                                                                            ▼
                                                              [BİLEŞİK DAMITMA KAYBI]
                                                              ├── Hard CE Loss: Gerçek Etiketler
                                                              ├── Soft KL Loss: Öğretmen Dağılımı
                                                              └── Sonuç       : 6.6x Hızlı & %93.6 Küçük
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Knowledge Distillation
Klasik Supervised Fine-Tuning (SFT) sadece one-hot doğru etiketleri ($\mathcal{L}_{\text{CE}}$) hedefler. Knowledge Distillation ise büyük modelin tüm kelime dağarcığı üzerindeki zengin olasılık dağılımını ($P_T$) hedef alarak bilgi transferi sağlar.

### 2. Yumuşatılmış Sıcaklık ($T$) ve Karanlık Bilgi (Dark Knowledge)
Standart softmax ($T=1$), doğru sınıfa %99 olasılık verirken diğer sınıfları sıfırlar. Sıcaklık $T > 1$ (örneğin $T=2.5$) yapıldığında:
$$P_i^T = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
İkincil ve üçüncül sınıflar arasındaki ince anlamsal benzerlikler (Dark Knowledge) görünür hale gelir ve öğrenciye aktarılır.

### 3. Monolithic Damıtma Kaybı Formülasyonu
Geoffrey Hinton tarafından önerilen standart damıtma kaybı:
$$\mathcal{L}_{\text{Total}} = \alpha \mathcal{L}_{\text{CE}}(y, P_S) + (1 - \alpha) T^2 D_{\text{KL}}(P_T^T \| P_S^T)$$
Buradaki $T^2$ çarpanı, sıcaklık arttıkça küçülen gradyan büyüklüğünü dengelemek için matematiksel olarak zorunludur.

### 4. Self-Instruct ve Endüstriyel Ölçek (DeepSeek-R1 Distill)
- **DeepSeek-R1-Distill-Qwen/Llama:** DeepSeek-R1'in 671B'lik modelinden üretilen 800K akıl yürütme verisi (CoT) kullanılarak 1.5B, 7B ve 8B modeller damıtılmıştır.
- **Parametre & Hız Tasarrufu:** Öğrenci model %93.6 daha az bellek harcar ve 6.6x daha yüksek çıkarım hızı (Throughput) sunar.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Knowledge Distillation** | Büyük bir öğretmen modelin bilgi ve yeteneklerini küçük bir öğrenci modele aktarma tekniği. |
| **Teacher Model** | Yüksek kapasiteli, geniş parametreli ve zengin temsil gücüne sahip kaynak model. |
| **Student Model** | Çıkarım hızı ve bellek tasarrufu için optimize edilmiş hedef küçük model. |
| **Dark Knowledge** | Modelin en yüksek tahmininin dışındaki ikincil sınıflarda saklı olan gizil anlamsal ilişkiler. |
| **Temperature ($T$)** | Softmax dağılımının düzgünlüğünü ve entropisini kontrol eden yumuşatma hiper-parametresi. |
| **KL Divergence ($D_{\text{KL}}$)** | Öğretmen ve öğrenci olasılık dağılımları arasındaki göreli entropi/uyumsuzluk ölçüsü. |
| **Hard Target** | Doğru sınıf için 1, diğerleri için 0 olan kesin zemin gerçeği (Ground Truth). |
| **Soft Target** | Öğretmen modelin sıcaklık ile yumuşatılmış sürekli olasılık dağılımı. |
| **Self-Instruct** | Öğretmen modelin kendi kendine talimat ve yanıt veri setleri üretmesi yöntemi. |
| **MiniLLM / Distill-R1** | Büyük akıl yürütme modellerinin küçük boyutlara damıtılmış açık kaynak varyantları. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %93.6 bellek tasarrufu.            │ • Küçük öğrencinin kapasite tavanı   │
 │ • 6.6x çıkarım hızlanması.           │   öğretmenin gerisinde kalabilir.    │
 │ • Karanlık bilgi aktarımı ile üstün  │ • Öğretmen modelin ileri yayılımı    │
 │   akıl yürütme (CoT) yeteneği.       │   eğitim sürecinde ek GPU gerektirir.│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Mobil, Edge ve IoT cihazlarda SOTA │ • Öğretmen halüsinasyon yaparsa      │
 │   zeka seviyesinde yerel LLM çalıştırma│  öğrenci de bu hatayı kopyalayabilir.│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/knowledge_distillation_paneli.png` dosyası üretilir:
1. **SFT vs KD Öğrenci Eğitim Kayıp Eğrisi (KD: 2.15 vs SFT: 6.88)**
2. **Öğretmen-Öğrenci Logit KL Diverjansı ($T^2 D_{\text{KL}}$)**
3. **Çıkarım Gecikmesi ve 6.6x Hızlanma Grafiği**
4. **Parametre Tasarrufu (%93.6 Küçülme)**
5. **Knowledge Distillation ve Self-Instruct Mimarisi Akış Şeması**
6. **Model Sıkıştırma ve Damıtma Kalite Sertifikası**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Knowledge Distillation ve model kıyaslama hattını koşturun
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
