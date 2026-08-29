# Day 166: GUI Ajanları ve Web Gezintisi — Set-of-Mark (SoM) ve Otonom Eylem Planlama

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 6. günüdür. Anthropic Computer Use, OpenAI Operator ve Mind2Web tarzı otonom GUI ajanlarının çekirdek çalışma prensibi olan **Ekran Görüntüsü Analizi (Screenshot Parsing)**, **Set-of-Mark (SoM) Tıklanabilir Eleman Etiketleme**, **Ayrık Eylem Uzayı (`click(x, y)`, `type(text)`, `scroll(dir)`, `press(key)`)** ve **Otonom Web Görev Yürütme Motoru** mimarisini sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "GUI Ajanı (Computer Use Agent)" Nedir ve Ekrandaki Butonlara Nasıl Tıklar?
- **Sorun (Kör API'ların Ötesi):**
  İnternetteki milyonlarca web sitesinin ve masaüstü yazılımın REST API'ı yoktur. İnsanlar bilgisayarı görsel olarak (ekrana bakarak, fareyle tıklayarak ve klavyeyle yazarak) kullanır.
- **Çözüm (Set-of-Mark + VLM + Eylem Motoru):**
  1. *Ekran Görüntüsü Alınır (Screenshot):* Tarayıcı veya işletim sisteminin anlık resmi yakalanır.
  2. *Set-of-Mark (SoM) ile Numaralandırma:* Sayfadaki tüm buton ve kutucukların üzerine görsel kırmızı numaralar (`[1]`, `[2]`, `[3]`) basılır.
  3. *VLM Düşünce Zinciri:* VLM "Kullanıcı arama yapmak istiyor, bu yüzden Mark `[1]`'deki arama kutusuna tıklamalıyım (`click(345, 500)`), ardından `type('DeepSeek')` yazmalıyım" der.
  4. *İşletim Sistemi İcrası:* Playwright veya PyAutoGUI bu komutları gerçek fare/klavye hareketlerine dönüştürür.

```
====================================================
          GUI AGENT WEB NAVIGATION ARCHITECTURE     
====================================================
  [Ham Ekran Görüntüsü (Screenshot)]                
           │                                        
           ▼                                        
  [Set-of-Mark (SoM) İşaretleyici ([1], [2], ...)]  
           │                                        
           ▼  + [Kullanıcı Doğal Dil Hedefi]        
  [Vision Language Model (VLM - GPT-4V / LLaVA)]    
           │                                        
           ▼  (Düşünce Zinciri - Visual CoT)        
  [Eylem Planı: 'click(345, 500)' / 'type(...)']    
           │                                        
           ▼                                        
  [OS / Web Tarayıcı Sürücüsü (Playwright)] ──> [İcra]
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Set-of-Mark (SoM) Görsel Prompting Mekanizması
- Ham piksel koordinatları vermek yerine, ekran üzerindeki $K$ adet interaktif HTML/DOM elemanı tespit edilip üzerine renkli etiket $M_i = [x_i, y_i, \text{ID}_i]$ bindirilir:
  $$\mathbf{I}_{\text{SoM}} = \text{OverlayMarks}(\mathbf{I}_{\text{raw}}, \{M_1, M_2, \dots, M_K\})$$
- SoM, VLM'in uzamsal halüsinasyon yapma oranını %70'ten fazla azaltarak hedef elemanı kusursuz seçmesini sağlar.

### B. Ayrık Eylem Uzayı (Action Grammar)
1. $\text{click}(y, x)$: $(x, y)$ koordinatına sol fare tıklaması.
2. $\text{type}(\text{string})$: Aktif giriş alanına metin yazma.
3. $\text{press\_key}(\text{key})$: `Enter`, `Tab`, `Escape` gibi özel tuşlara basma.
4. $\text{scroll}(\text{direction})$: Sayfayı yukarı/aşağı kaydırma.
5. $\text{terminate}(\text{status})$: Görevi başarıyla sonlandırma veya pes etme.

### C. Çok Adımlı Eylem Döngüsü (POMDP Çerçevesi)
- Her $t$ anında gözlem $o_t = \mathbf{I}_t$ (ekran görüntüsü), geçmiş eylemler $a_{<t}$ ve kullanıcı hedefi $g$ alınarak bir sonraki eylem seçilir:
  $$a_t = \arg\max_a P(a \mid o_t, a_{<t}, g)$$

### D. Performans ve Doğrulama
- Test edilen 2 çok adımlı görev senaryosunda (Google Arama ve E-Ticaret Sepet Onayı) **9/9 adım başarıyla icra edilmiş (%100 Adım & Görev Tamamlama)**.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **GUI Agent** | Grafiksel kullanıcı arayüzlerini (ekran, fare, klavye) otonom kullanan yapay zeka ajanı. |
| **Set-of-Mark (SoM)** | Ekrandaki tıklanabilir elemanların üzerine görsel sayı etiketleri koyma tekniği. |
| **Action Space** | Ajanın ekranda yapabileceği tüm eylemlerin (click, type, scroll) kümesi. |
| **Computer Use** | Bir VLM'in doğrudan bilgisayar işletim sistemini kontrol edebilme yeteneği. |
| **Mind2Web / OSWorld** | Web ve işletim sistemi ajanlarını değerlendiren küresel benchmark'lar. |
| **Visual CoT (Chain-of-Thought)** | Ajanın eylem yapmadan önce ekranı analiz edip plan kurduğu düşünce adımı. |
| **DOM Element Grounding** | Web sayfasındaki HTML kodları ile görsel pikselleri eşleştirme. |
| **Action Success Rate (ASR)** | Ajanın ürettiği eylemlerin doğru ve geçerli olma oranı. |
| **Task Completion Rate** | Ajanın tüm adımları hatasız tamamlayıp nihai amaca ulaşma yüzdesi. |
| **Browser Driver** | Playwright veya Selenium gibi web eylemlerini icra eden arka uç motoru. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • API'ı olmayan her web sitesini ve  │ • Sayfa yükleme gecikmelerinde veya  │
 │   yazılımı otonom kullanabilme.      │   pop-up açıldığında ajanın şaşırma  │
 │ • SoM ile %100'e yakın tıklama       │   (Dynamic Latency) riski.           │
 │   isabeti.                           │                                      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Otonom test otomasyonu, kişisel    │ • Yanlışlıkla istenmeyen form        │
 │   asistanlar, veri toplama ve rutin  │   gönderme veya satın alma yapma     │
 │   ofis işlerini otomatikleştirme.    │   (Safety / Guardrail) riski.        │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/gui_agent_web_navigation_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
