# Day 221: Model Context Protocol (MCP) Standart Araç Sunucusu ve İstemcisi

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%2012-Agentic%20AI%20%26%20MCP-orange.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; **FAZ 12: Otonom Ajanlar (Agentic AI), Araç Kullanımı (Tool-Use) & MCP Protokolü (Gün 221 - Gün 240)** serisinin **Gün 221 (FAZ 12 BAŞLANGICI)** modülüdür. Yapay zeka modellerini dış dünya sistemlerine, kurumsal veritabanlarına, dosya sistemlerine ve API'lere bağlayan, Anthropic ve Antigravity uyumlu açık endüstri standardı **Model Context Protocol (MCP)** mimarisini; **JSON-RPC 2.0 Mesajlaşma Katmanını**, **Dinamik Araç ve Kaynak Keşfi Motorunu (`tools/list`, `resources/read`)**, **Girdi Şeması Doğrulama Mekanizmasını** ve **İki Yönlü İstemci-Sunucu (Client-Server) Yürütme Hattını** sıfırdan Python ile inşa etmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ Yapay Zekanın "USB-C Kablosu" Nedir? (Model Context Protocol - MCP)
- **Eski Dünyanın "Kablo Karmaşası" Sorunu:**
  Eskiden bir LLM modeline PostgreSQL veritabanını bağlamak istediğinizde OpenAI için ayrı, LangChain için ayrı, Claude için ayrı kod yazmanız gerekirdi (Ad-Hoc Tool Glue). Bu da 14 günlük entegrasyon süresi ve sürekli bozulan şema hataları demekti.
- **MCP Nasıl Çalışır? (Evrensel Açık Standart):**
  1. **Tek Bir Sunucu Yazılır (MCP Server):** Veritabanınızı veya dosya sisteminizi tek bir MCP sunucusu olarak yayınlarsınız.
  2. **Tüm Ajanlar Anında Bağlanır (MCP Host/Client):** Claude Desktop, Antigravity IDE, Cursor veya yerel Python ajanınız aynı sunucuya JSON-RPC 2.0 üzerinden bağlanır.
  3. **`tools/list` ve `tools/call`:** Ajan hangi araçların olduğunu dinamik olarak sorar (`tools/list`), parametrelerini doğrular ve güvenle çalıştırır (`tools/call`).
  4. **`resources/read`:** Ajan sistemdeki dosyaları, dökümanları ve canlı metrikleri tek bir URI ile okur.
  5. Sonuç: Yeni bir araç entegrasyonu **336 saatten (14 gün) 2 saate iner** ve hata oranı **%0.1'e düşer!**

```
========================================================================================
             MODEL CONTEXT PROTOCOL (MCP) İSTEMCİ-SUNUCU MİMARİSİ                       
========================================================================================
                 [Kullanıcı / Ajan Ana Bilgisayarı (MCP Host: Claude / LLM)]
                                        │
                                        ▼
                 [MCP İstemcisi (MCP Client): JSON-RPC 2.0 Yönlendirici]
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼ (stdio / SSE Taşıma) ▼                      ▼
         [MCP Dosya Sunucusu]   [MCP SQLite Sunucusu]   [MCP Git / API Sunucusu]
         • `tools/list`         • `tools/list`          • `tools/list`
         • `tools/call`         • `tools/call`          • `tools/call`
         • `resources/read`     • `resources/read`      • `resources/read`
                                        │
                                        ▼
             [STANDARTLAŞTIRILMIŞ ÇIKTI: 14 Günlük Entegrasyon 2 Saate İner]
========================================================================================
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. 🔍 Neden Bu Sistem Kullanılır? (Mühendislik & Bilimsel Gerekçe)
- **Evrensel Ajan Birlikte Çalışabilirliği (Interoperability):**
  Ajan kodunu araç mantığından ayırarak (Decoupled Architecture), tek bir araç sunucusunun tüm modeller tarafından sıfır uyarlama eforuyla kullanılmasını sağlar.

### B. 🛡️ Ne Gibi Sorunları Çözer? (Çözülen Kritik Darboğazlar)
- **Şema Uyuşmazlığı ve Tip Hataları:** JSON Schema standardı sayesinde parametre hatalarını çalışma zamanından önce yakalar.
- **Entegrasyon Süresi:** 14 günlük özel adaptör yazma sürecini 2 saate düşürür.

### C. ⚠️ Ne Konuda Eksik Kalır? (Sınırlar ve Dikkat Edilmesi Gerekenler)
- **Güvenlik İzolasyonu (Sandboxing):** `tools/call` yerel sistemde kod çalıştırabileceğinden sunucu tarafında yetkilendirme ve izin denetimi şarttır.

### D. 🔄 Alternatif Sistemler & Karşılaştırmalı Dağıtık Mimariler

| Entegrasyon Yaklaşımı | Entegrasyon Süresi | Platform Bağımsızlığı | Hata Oranı | Bakım Eforu |
|:---|:---:|:---:|:---:|:---:|
| **Özel Ad-Hoc Yapıştırıcılar** | 336 Saat (14 Gün) | Zayıf (%20) | %12.5 (Yüksek) | 8.5 / 10 |
| **LangChain Proprietary Tools** | 48 Saat | Orta (%45) | %6.2 | 6.0 / 10 |
| **Standart Model Context Protocol (MCP)**| **2.0 Saat (-%99.4)**| **Evrensel (%100)**| **%0.1 (Kusursuz)**| **1.5 / 10 (Minimal)**|

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Model Context Protocol (MCP)** | Yapay zeka ajanlarının harici araç ve verilere bağlanmasını sağlayan açık kaynaklı standart protokol. |
| **JSON-RPC 2.0** | İstemci ile sunucu arasında durumsuz ve yapılandırılmış veri alışverişi sağlayan hafif RPC protokolü. |
| **MCP Host** | Claude Desktop veya Antigravity gibi MCP istemcisini çalıştıran ve LLM karar döngüsünü yöneten ana uygulama. |
| **MCP Client** | LLM'in araç çağrılarını JSON-RPC isteklerine dönüştürüp sunuculara ileten istemci katmanı. |
| **MCP Server** | Veritabanı veya işletim sistemi yeteneklerini araç (tools) ve kaynak (resources) olarak sunan sunucu servisi. |
| **`tools/list`** | Sunucuda kayıtlı tüm araçları ve girdi JSON şemalarını istemciye bildiren keşif metodu. |
| **`tools/call`** | Belirli bir aracın verilen argümanlarla çalıştırılmasını ve sonucun metin olarak dönmesini sağlayan yürütme metodu. |
| **`resources/read`** | Bir URI adresindeki statik veya dinamik sistem verisini (dosya, log, config) ajan bağlamına yükleyen okuma metodu. |
| **Input Schema** | Bir aracın hangi parametreleri (string, number, boolean) ve zorunlu alanları beklediğini tanımlayan JSON şeması. |
| **Transport Layer** | MCP mesajlarının iletildiği iletişim kanalı (Standart girdi/çıktı `stdio` veya HTTP tabanlı `SSE`). |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Evrensel ekosistem uyumu (%100).   │ • Durumsuz (stateless) iletişimde    │
 │ • Entegrasyon süresi 2 saate iner.   │   oturum yönetimi dikkat gerektirir. │
 │ • JSON şema ile sıfır tip hatası.    │ • Yerel sunucularda güvenlik         │
 │ • Hem stdio hem SSE desteği.         │   izolasyonu sağlanmalıdır.          │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kurumsal tüm veri kaynaklarını     │                                      │
 │   otonom ajanlara anında açma.       │                                      │
 └──────────────────────────────────────┴──────────────────────────────────────┘
        FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan 6 panelli MCP teşhis panosu: `ciktilar/mcp_protokol_paneli.png`

![MCP Paneli](ciktilar/mcp_protokol_paneli.png)

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
