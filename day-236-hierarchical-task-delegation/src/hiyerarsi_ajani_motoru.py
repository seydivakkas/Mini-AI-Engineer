"""
Hiyerarşik Görev Delegasyonu Motoru (Day 236 - FAZ 12).
Kök Yönetici (Manager) ve Uzman İşçiler (Workers) Arasında Yük Paylaşımı.
"""

from typing import Dict, Any, List, Optional


class SubTask:
    """Ayrıştırılmış Alt Görev Modeli (WBS)."""

    def __init__(self, task_id: int, alan: str, aciklama: str):
        self.task_id = task_id
        self.alan = alan.lower()
        self.aciklama = aciklama
        self.durum = "BEKLIYOR"
        self.cikti: Optional[str] = None


class WorkerAgent:
    """Belirli Bir Alanda Uzmanlaşmış İşçi Ajan."""

    def __init__(self, alan: str, uzmanlik: str):
        self.alan = alan.lower()
        self.uzmanlik = uzmanlik

    def gorev_yurut(self, task: SubTask) -> str:
        """Uzmanlık alanına göre alt görevi icra eder."""
        task.durum = "YURUTULUYOR"

        if self.alan == "database":
            sonuc = f"PostgreSQL 'users' tablosu ve 'email_idx' indeksi başarıyla oluşturuldu."
        elif self.alan == "backend":
            sonuc = f"FastAPI '/auth/login' ve '/auth/register' JWT endpointleri hazırlandı."
        elif self.alan == "security":
            sonuc = f"Bcrypt şifreleme ve 10 req/min IP Rate Limiting güvenlik denetiminden geçti."
        else:
            sonuc = f"'{task.aciklama}' genel görevi tamamlandı."

        task.durum = "TAMAMLANDI"
        task.cikti = sonuc
        return sonuc


class ManagerAgent:
    """Kök Hedefi Ayrıştıran, Delege Eden ve Sonuçları Sentezleyen Yönetici Ajan."""

    def __init__(self):
        self.isciler: Dict[str, WorkerAgent] = {}
        self.islem_gunlugu: List[str] = []

    def isci_kaydet(self, isci: WorkerAgent):
        self.isciler[isci.alan] = isci

    def gorevi_ayristir(self, kok_hedef: str) -> List[SubTask]:
        """Kök hedefi uzmanlık alanlarına göre alt görevlere (WBS) böler."""
        return [
            SubTask(1, "database", "Kullanıcı kimlik doğrulama veritabanı şemasını kur."),
            SubTask(2, "backend", "JWT token üreten ve doğrulayan REST API rotalarını yaz."),
            SubTask(3, "security", "Bcrypt hashleme ve Rate Limiting güvenlik kontrollerini yap."),
        ]

    def gorevleri_delege_et_ve_birlestir(self, kok_hedef: str) -> Dict[str, Any]:
        """Alt görevleri ilgili uzman işçilere dağıtır ve nihai sentez raporu üretir."""
        alt_gorevler = self.gorevi_ayristir(kok_hedef)
        self.islem_gunlugu.append(f"Yönetici kök hedefi {len(alt_gorevler)} alt göreve ayrıştırdı.")

        tamamlanan_cikti: Dict[str, str] = {}

        for task in alt_gorevler:
            isci = self.isciler.get(task.alan)
            if isci:
                self.islem_gunlugu.append(
                    f"Görev #{task.task_id} [{task.alan.upper()}] -> {isci.uzmanlik} işçisine devredildi."
                )
                cikti = isci.gorev_yurut(task)
                tamamlanan_cikti[task.alan] = cikti
                self.islem_gunlugu.append(f"Görev #{task.task_id} tamamlandı: {cikti}")
            else:
                task.durum = "BASARISIZ"
                self.islem_gunlugu.append(f"Görev #{task.task_id} için uygun işçi bulunamadı!")

        # Nihai Sentez Raporu
        sentez = (
            f"Kök Hedef ('{kok_hedef}') başarıyla tamamlandı. "
            f"Veritabanı, Backend API ve Güvenlik katmanları entegre edildi."
        )

        return {
            "kok_hedef": kok_hedef,
            "alt_gorevler": alt_gorevler,
            "tamamlanan_ciktilar": tamamlanan_cikti,
            "sentez": sentez,
            "gunluk": self.islem_gunlugu,
        }
