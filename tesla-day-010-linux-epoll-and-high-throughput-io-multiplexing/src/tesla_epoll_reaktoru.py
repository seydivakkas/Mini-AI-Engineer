"""
Tesla Linux epoll ve Olay Tabanli Coklama (I/O Multiplexing) Reaktoru
====================================================================
Bu modul; Linux kernel `epoll_create1`, `epoll_ctl` ve `epoll_wait` API'lerini,
Edge-Triggered (`EPOLLET`) modunu ve $O(1)$ olceklenme performansini gercekler.
8 kamera video akisi ve 4 CAN veri yolu soketini tek bir reaktor dongusunde yonetir.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import time


class EpollTetiklemeModu(Enum):
    LEVEL_TRIGGERED = "LEVEL_TRIGGERED"      # Standart: Tampon bosalana kadar her epoll_wait'te tetiklenir
    EDGE_TRIGGERED_EPOLLET = "EPOLLET"       # Hard RT: Yalnizca yeni veri geldiginde 1 kez tetiklenir (Non-blocking zorunlu)


class EpollOlayTipi:
    EPOLLIN = 0x001   # Okunabilir veri var
    EPOLLOUT = 0x004  # Yazmaya hazir
    EPOLLERR = 0x008  # Hata durumu
    EPOLLHUP = 0x010  # Baglanti koptu


@dataclass
class EpollKayitliDosya:
    fd_id: int
    olay_maskesi: int
    tetikleme_modu: EpollTetiklemeModu
    kullanici_verisi: Any
    tampondaki_bayt_sayisi: int = 0
    yeni_olay_geldi_mi: bool = False


class TeslaEpollOlayReaktoru:
    """
    Linux epoll O(1) Reaktör Döngüsü.
    """
    def __init__(self):
        self.kayitli_fdler: Dict[int, EpollKayitliDosya] = {}
        self.hazir_olay_kuyrugu: Set[int] = set()

    def epoll_ctl_ekle(self, fd_id: int, olay_maskesi: int, kullanici_verisi: Any,
                       tetikleme: EpollTetiklemeModu = EpollTetiklemeModu.EDGE_TRIGGERED_EPOLLET) -> bool:
        """`epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &event)` çağrısı."""
        if fd_id in self.kayitli_fdler:
            return False
        self.kayitli_fdler[fd_id] = EpollKayitliDosya(
            fd_id=fd_id,
            olay_maskesi=olay_maskesi,
            tetikleme_modu=tetikleme,
            kullanici_verisi=kullanici_verisi
        )
        return True

    def epoll_ctl_sil(self, fd_id: int) -> bool:
        """`epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL)` çağrısı."""
        if fd_id in self.kayitli_fdler:
            del self.kayitli_fdler[fd_id]
            self.hazir_olay_kuyrugu.discard(fd_id)
            return True
        return False

    def veri_geldi_sinyali(self, fd_id: int, bayt_sayisi: int):
        """Donanımsal kesme veya sokete veri gelmesi durumunda kernel'in epoll listesini güncellemesi."""
        if fd_id in self.kayitli_fdler:
            kayit = self.kayitli_fdler[fd_id]
            kayit.tampondaki_bayt_sayisi += bayt_sayisi
            kayit.yeni_olay_geldi_mi = True
            self.hazir_olay_kuyrugu.add(fd_id)

    def epoll_wait(self, maks_olay: int = 64) -> List[Dict[str, Any]]:
        """
        `epoll_wait(epfd, events, maxevents, timeout)`
        Hazır olan dosya tanımlayıcıları O(1) karmaşıklıkla döndürür.
        """
        tetiklenen_olaylar: List[Dict[str, Any]] = []
        hazir_listesi = list(self.hazir_olay_kuyrugu)[:maks_olay]

        for fd_id in hazir_listesi:
            kayit = self.kayitli_fdler[fd_id]
            
            tetiklenen_olaylar.append({
                "fd_id": fd_id,
                "olaylar": EpollOlayTipi.EPOLLIN,
                "veri": kayit.kullanici_verisi,
                "tampon_boyutu": kayit.tampondaki_bayt_sayisi
            })

            if kayit.tetikleme_modu == EpollTetiklemeModu.EDGE_TRIGGERED_EPOLLET:
                # EPOLLET modunda 1 kez bildirilir, olay kuyruktan çıkar
                kayit.yeni_olay_geldi_mi = False
                self.hazir_olay_kuyrugu.discard(fd_id)
            else:
                # Level-triggered modunda tamponda veri olduğu sürece kuyrukta kalır
                if kayit.tampondaki_bayt_sayisi <= 0:
                    self.hazir_olay_kuyrugu.discard(fd_id)

        return tetiklenen_olaylar

    def tamponu_bosalt_tuket(self, fd_id: int):
        """Non-blocking okuma ile EAGAIN/EWOULDBLOCK alana kadar tamponu tamamen boşaltma."""
        if fd_id in self.kayitli_fdler:
            self.kayitli_fdler[fd_id].tampondaki_bayt_sayisi = 0
            self.hazir_olay_kuyrugu.discard(fd_id)


class TeslaOlayFd:
    """
    Linux `eventfd(0, EFD_NONBLOCK)` çekirdekler arası hafif sinyalleşme mekanizması.
    """
    def __init__(self, baslangic_sayaci: int = 0):
        self.sayac = baslangic_sayaci

    def sinyal_yaz(self, deger: int = 1):
        """`write(eventfd, &val, 8)`"""
        self.sayac += deger

    def sinyal_oku(self) -> int:
        """`read(eventfd, &val, 8)` - Sayaç okunup sıfırlanır."""
        val = self.sayac
        self.sayac = 0
        return val
