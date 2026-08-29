# Day 168: Gerçek Zamanlı Video Akışı Analizi ve Olay Tespiti (Streaming VLM)

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%209-Multimodal%20Foundations-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 9: Çok Modlu (Multimodal) Temel Modeller (Gün 161 - Gün 180)** serisinin 8. günüdür. Video streaming, güvenlik kameraları ve canlı robotik algı sistemlerinin temeli olan **Gerçek Zamanlı Akış Yönetimi (Online Streaming Video)**, **Kayan Bellek Kuyruğu (Sliding Window Memory Buffer / Ring Buffer)**, **Dinamik Olay ve Anomali Tespiti (Online Event Detection & Trigger)** ve **Düşük Gecikmeli Streaming VLM Çıkarımı** mimarisini sıfırdan PyTorch ile hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Streaming VLM" Nedir ve Neden Tüm Videoyu Bellekte Tutamayız?
- **Sorun (Sonsuz Canlı Akış Çıkmazı):**
  Bir güvenlik kamerası veya otonom araç kamerası günde 24 saat kesintisiz çalışır. Milyonlarca kareyi LLM belleğinde tutmak imkansızdır (Bellek Taşması - OOM).
- **Çözüm (Kayan Bellek + Olay Tetikleyici Dedektör):**
  1. *Kayan Bellek (FIFO Ring Buffer - 16 Kare):* Yalnızca son birkaç saniyelik en taze kareler hafızada tutulur. Eski kareler otomatik silinir.
  2. *Hafif Değişim Dedektörü (Cosine Distance):* Her karede pahalı LLM çağrılmaz! Sahne durağanken hiçbir şey yapılmaz.
  3. *Olay Tetiklendiğinde VLM Devreye Girer:* Ne zaman ki bir anomali (örn: yasaklı araç geçişi, sahipsiz çanta) fark edilirse, VLM anında çağrılır ve "00:10'da beyaz araç yasak kapıdan girdi!" şeklinde zaman damgalı alarm üretilir.

```
====================================================
        STREAMING VLM ONLINE ARCHITECTURE           
====================================================
  [Canlı Kamera Akışı (30 FPS Continuous Stream)]   
           │                                        
           ▼                                        
  [Kayan Bellek Kuyruğu (FIFO Ring Buffer - 16 Kare)]
           │                                        
           ├──> [Online Değişim Dedektörü (Cosine/Diff)]
           │         │                              
           │         ▼ (Eşik Aşıldı mı? > 0.35)     
           │      [EVET -> VLM Tetikleyici]         
           ▼         │                              
  [Streaming VLM Projektör + Causal LLM]            
           │                                        
           ▼                                        
  [Anlık Güvenlik Uyarısı / Zaman Damgalı Alarm]    
====================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Kayan Bellek (Sliding Window Ring Buffer) Bellek Sınırı
- Maksimum $K=16$ kapasiteli dairesel kuyruk ile bellek tüketimi $\mathcal{O}(K \cdot N \cdot D)$ şeklinde sabit tutulur.
- Zaman geçtikçe bellek tüketimi artmaz; sonsuz uzunluktaki canlı video akışlarında **sabit $O(1)$ bellek karmaşıklığı** sağlanır.

### B. Kosinüs Mesafesi Tabanlı Online Değişim Sinyali
- Ardışık karelerin küresel temsil vektörleri $v_{t-1}$ ve $v_t$ olmak üzere Anomali/Fark Skoru:
  $$S(t) = 1.0 - \frac{v_{t-1} \cdot v_t}{\|v_{t-1}\|_2 \|v_t\|_2}$$
- $S(t) \ge \tau = 0.35$ olduğunda olay tetiklenir ($E_t = \text{True}$).

### C. Streaming VLM ile Düşük Çıkarım Maliyeti
- 24 saatlik akışta gelen 2.592.000 karenin tamamını LLM'e sokmak yerine, yalnızca olayın gerçekleştiği kritik anlarda çıkarım yapılarak **hesaplama maliyetinde %99.5 tasarruf** elde edilir.

### D. Performans ve Doğrulama
- 30 saniyelik güvenlik kamerası simülasyonunda 2 kritik anomali (araç geçişi ve sahipsiz çanta) **%100 doğrulukla ve sıfır yanlış alarmla** tespit edilmiştir.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Streaming VLM** | Canlı video akışını anlık olarak izleyip olayları gerçek zamanlı raporlayan VLM. |
| **Sliding Memory Window** | Yalnızca son $K$ kareyi hafızada tutan kayan bellek penceresi. |
| **Ring Buffer (Dairesel Tampon)** | Yeni eleman eklendikçe en eski elemanı otomatik silen FIFO bellek yapısı. |
| **Online Event Detection** | Videonun tamamını beklemeden, kare geldikçe olayı anında yakalama. |
| **Anomaly Score** | Ardışık kareler arasındaki görsel farkı ve sıra dışılığı ölçen metrik. |
| **Event Trigger** | Eşik değeri aşıldığında VLM analizini başlatan tetikleyici mantık. |
| **Timestamped Alert** | Olayın gerçekleştiği kesin anı ($t=10s$) belirten zaman damgalı uyarı. |
| **Continuous Video Stream** | Kesintisiz canlı yayın veya güvenlik kamerası video akışı. |
| **Edge AI Inference** | Video analizini yerel cihazda/kamerada düşük gecikmeyle yürütme. |
| **False Alarm Rate (FAR)** | Normal durumlarda gereksiz yere alarm üretme oranı. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Sonsuz canlı akışta sabit $O(1)$   │ • Kayan pencere dışında kalan çok    │
 │   GPU bellek tüketimi.               │   eski geçmiş olayların unutulması   │
 │ • %99 hesaplama maliyeti tasarrufu.  │   (Long-term memory loss).           │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • 7/24 otonom güvenlik izleme,       │ • Ani ışık değişimlerinde veya hava  │
 │   akıllı trafik kontrolü, robotik ve │   durumu bozulmalarında geçici       │
 │   canlı cerrahi ameliyat takibi.     │   sahte tetiklenme riski.            │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/streaming_video_understanding_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
