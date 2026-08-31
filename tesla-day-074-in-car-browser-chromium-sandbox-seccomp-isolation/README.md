# 🚗 Tesla FSD Otonom Sürüş | Gün 74: Araç İçi Web Tarayıcısı, Chromium Sandbox ve Güvenlik İzolasyonu

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Chromium](https://img.shields.io/badge/Browser-Chromium%20Embedded%20Framework-red.svg?style=flat-square)](https://www.tesla.com/)
[![Seccomp-BPF](https://img.shields.io/badge/Isolation-Linux%20Seccomp--BPF%20Sandbox-blue.svg?style=flat-square)](https://man7.org/linux/man-pages/man2/seccomp.2.html)
[![Zero-Trust](https://img.shields.io/badge/Security-Zero--Trust%20CAN--Bus%20Shield-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"74. günümüze hoş geldin stajyer!  
> Tesla'nın dokunmatik ekranında YouTube izleyebilir, Netflix açabilir veya web sitelerinde gezinebilirsiniz. Ancak bir hacker kötü amaçlı bir web sayfası (Drive-by Download / WebKit 0-day) üzerinden aracın direksiyonuna veya frenlerine erişebilir mi?  
> Asla! Çünkü Tesla **Sıfır Güven (Zero Trust) Seccomp-BPF Chromium Kum Havuzu (Sandbox)** mimarisini uygular:  
> 1. **Sistem Çağrısı (Syscall) Filtreleme:** Linux çekirdeği seviyesinde Seccomp-BPF filtresi çalışır. Tarayıcı yalnızca grafik çizme ve temel bellek çağrıları (`read`, `write`, `mmap`) yapabilir.  
> 2. **Kritik Çağrıların Engellenmesi:** Tarayıcı asla ham ağ soketi (`socket`), süreç izleme (`ptrace`), yeniden başlatma (`reboot`) veya çekirdek modülü yükleme (`bpf`) yapamaz.  
> 3. **CAN-Bus İzolasyonu:** Web tarayıcısı süreci araç içi CAN ağına tamamen sağırdır ve kördür; hiçbir donanım adresine erişemez.  
> 4. **Anında Sonlandırma (`SIGSYS`):** Yasaklı tek bir çağrı yapıldığı anda tarayıcı sekmesi çekirdek tarafından anında imha edilir.  
> Bugün Tesla'nın araç içi eğlence ekranını siber saldırılardan koruyan Seccomp Sandbox kalkanını kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Seccomp-BPF Sistem Çağrısı Karar Fonksiyonu

$$\text{Action}(\text{Syscall}) = \begin{cases} \text{SECCOMP\_RET\_ALLOW}, & \text{Syscall} \in \mathcal{S}_{\text{allowed}} \ (\text{read, write, mmap, futex}) \\ \text{SECCOMP\_RET\_KILL\_PROCESS}, & \text{Syscall} \in \mathcal{S}_{\text{blocked}} \ (\text{socket, ptrace, reboot, bpf}) \end{cases}$$

### 2. Zero Trust Kum Havuzu İzolasyon Olasılığı

$$P(\text{CAN\_Bus\_Compromise} \mid \text{Browser\_Exploited}) = 0.0000$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Web sayfalarındaki güvensiz JavaScript kodları ve tarayıcı açıklarının (0-day) araç sürüş ve kontrol sistemlerine sızmasını engellemek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Kum Havuzundan Kaçış (Sandbox Escape):** Çekirdek düzeyinde BPF filtreleri ile tarayıcının root yetkisi almasını veya dosya sistemini kurcalamasını engelledi.
- **CAN-Bus İhlali:** Tarayıcının SocketCAN donanımına doğrudan mesaj enjekte etmesini imkansız kıldı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Gelişmiş Web Özellikleri:** Çok katı Seccomp filtreleri bazı gelişmiş WebGL/WebGPU veya yerel donanım erişimi gerektiren özellikleri kısıtlayabilir.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Korumasız Tarayıcı (Raw WebKit):** Otomotivde kullanılmaz; doğrudan ölümcül siber saldırılara kapı açar.
- **Kapsayıcı / Docker (Containerization):** İyidir ancak Seccomp-BPF kadar hafif ve çekirdek seviyesinde hızlı değildir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Chromium Sandbox** | Web içeriğini render eden süreci kısıtlı yetkilerle çalıştıran güvenlik kafesi. |
| **Seccomp-BPF** | Linux çekirdeğinde süreçlerin yapabileceği sistem çağrılarını Berkeley Packet Filter ile denetleyen mekanizma. |
| **Syscall (System Call)** | Bir kullanıcı süreci ile işletim sistemi çekirdeği arasındaki fonksiyon çağrısı. |
| **Zero Trust** | Kullanıcı arayüzündeki hiçbir üçüncü parti koda veya web sitesine güvenmeme felsefesi. |
| **Sandbox Escape** | Kötü niyetli bir kodun kum havuzundan çıkarak ana işletim sisteminde yetki yükseltmesi. |
| **ptrace** | Bir sürecin başka bir sürecin belleğini ve CPU yazmaçlarını izlemesini sağlayan (yasaklı) çağrı. |
| **SocketCAN** | Linux çekirdeğinin CAN-Bus mesajlarını ağ soketi olarak açtığı arayüz (tarayıcıya yasaklı). |
| **SIGSYS** | Seccomp kuralını ihlal eden bir sürece çekirdek tarafından gönderilen ölümcül sinyal. |
| **mmap / munmap** | Sayfa düzeyinde bellek tahsisi ve serbest bırakma sistem çağrıları (izinli). |
| **Zero-Day Exploit** | Yazılım geliştiricisi tarafından henüz bilinmeyen veya yaması çıkmamış güvenlik açığı. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • Linux çekirdeği seviyesinde %100 aşılmaz izolasyon  | • Çok katı filtrelerin bazı WebUSB vb. çevre birim    |
| • 0.3 µs ultra hızlı çağrı kontrol gecikmesi          |   erişimlerini engellemesi                            |
| • Sıfır CAN-Bus ve araç donanımı sızma riski          |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Araç içi web mağazası ve üçüncü parti güvenli       | • Linux çekirdeğinin kendisinde çıkabilecek nadir     |
|   uygulama ekosisteminin temeli                       |   ayrıcalık yükseltme (Privilege Escalation) açıkları |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Chromium Seccomp Sandbox Güvenlik Şeması

```
[ İnternet Web Sayfası (YouTube / Web Siteleri) ]
                       |
                       v
       [ Chromium Renderer (Kullanıcı Alanı) ]
                       |
                       | Sistem Çağrısı (Syscall) Talebi
                       v
       [ Linux Seccomp-BPF Çekirdek Filtresi ]
            |                             |
            | İzinli: read, write, mmap    | Yasaklı: socket, ptrace, reboot
            v                             v
[ İşletim Sistemi İcrası Yapılır ]   [ SÜREÇ ANINDA ÖLDÜRÜLÜR (SIGSYS) ]
                                     - CAN-Bus %100 Korundu
                                     - Sıfır Siber Saldırı Riski
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Chromium Sandbox simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
