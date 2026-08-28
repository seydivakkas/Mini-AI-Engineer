"""
Merkezi Deney Takibi ve Artefakt Kayıt Motoru (MLflow / W&B Mimarisi)
---------------------------------------------------------------------
Deney yönetimi (Experiments), Koşu yaşam döngüsü (Runs), Parametre & Zaman Serisi
Metrik Kaydı, Model Ağırlığı ve Artefakt Depolama Sistemi.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import os
import sys
import time
import json
import uuid
import shutil
import sqlite3
import torch


class DeneyKosusu:
    """
    Tek bir model eğitim koşusunu (Run) temsil eden ve loglama işlemlerini yürüten sınıf.
    """
    def __init__(
        self,
        run_id: str,
        experiment_id: str,
        experiment_name: str,
        depo_dizini: str,
        baglanti: sqlite3.Connection
    ):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.depo_dizini = depo_dizini
        self.baglanti = baglanti

        self.parametreler: Dict[str, Any] = {}
        self.etiketler: Dict[str, str] = {}
        self.metrik_gecmisi: Dict[str, List[Dict[str, Any]]] = {}
        self.son_metrikler: Dict[str, float] = {}
        self.artefakt_dizini = os.path.join(self.depo_dizini, "artefaktlar", self.experiment_id, self.run_id)
        os.makedirs(self.artefakt_dizini, exist_ok=True)

    def log_param(self, anahtar: str, deger: Any) -> None:
        self.parametreler[anahtar] = deger
        deger_str = json.dumps(deger) if isinstance(deger, (dict, list)) else str(deger)
        with self.baglanti:
            self.baglanti.execute(
                "INSERT OR REPLACE INTO parametreler (run_id, anahtar, deger) VALUES (?, ?, ?)",
                (self.run_id, anahtar, deger_str)
            )

    def log_params(self, param_sozlugu: Dict[str, Any]) -> None:
        for k, v in param_sozlugu.items():
            self.log_param(k, v)

    def log_metric(self, anahtar: str, deger: float, step: Optional[int] = None) -> None:
        if anahtar not in self.metrik_gecmisi:
            self.metrik_gecmisi[anahtar] = []

        kayit = {
            "step": step if step is not None else len(self.metrik_gecmisi[anahtar]),
            "value": float(deger),
            "timestamp": time.time()
        }
        self.metrik_gecmisi[anahtar].append(kayit)
        self.son_metrikler[anahtar] = float(deger)

        with self.baglanti:
            self.baglanti.execute(
                "INSERT INTO metrikler (run_id, anahtar, deger, step, timestamp) VALUES (?, ?, ?, ?, ?)",
                (self.run_id, anahtar, float(deger), kayit["step"], kayit["timestamp"])
            )

    def log_metrics(self, metrik_sozlugu: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrik_sozlugu.items():
            self.log_metric(k, v, step=step)

    def log_artifact(self, yerel_dosya_yolu: str, hedef_alt_dizin: Optional[str] = None) -> str:
        if not os.path.exists(yerel_dosya_yolu):
            raise FileNotFoundError(f"Kaydedilecek artefakt bulunamadı: {yerel_dosya_yolu}")

        hedef_dizin = self.artefakt_dizini
        if hedef_alt_dizin:
            hedef_dizin = os.path.join(hedef_dizin, hedef_alt_dizin)
            os.makedirs(hedef_dizin, exist_ok=True)

        dosya_adi = os.path.basename(yerel_dosya_yolu)
        hedef_tam_yol = os.path.join(hedef_dizin, dosya_adi)

        if os.path.abspath(yerel_dosya_yolu) != os.path.abspath(hedef_tam_yol):
            if os.path.isdir(yerel_dosya_yolu):
                if os.path.exists(hedef_tam_yol):
                    shutil.rmtree(hedef_tam_yol)
                shutil.copytree(yerel_dosya_yolu, hedef_tam_yol)
            else:
                shutil.copy2(yerel_dosya_yolu, hedef_tam_yol)

        with self.baglanti:
            self.baglanti.execute(
                "INSERT OR REPLACE INTO artefaktlar (run_id, dosya_adi, dosya_yolu) VALUES (?, ?, ?)",
                (self.run_id, dosya_adi, hedef_tam_yol)
            )
        return hedef_tam_yol

    def log_model(self, model: torch.nn.Module, model_adi: str = "model.pt") -> str:
        hedef_yol = os.path.join(self.artefakt_dizini, model_adi)
        torch.save(model.state_dict(), hedef_yol)
        return self.log_artifact(hedef_yol)

    def set_tag(self, anahtar: str, deger: str) -> None:
        self.etiketler[anahtar] = str(deger)
        with self.baglanti:
            self.baglanti.execute(
                "INSERT OR REPLACE INTO etiketler (run_id, anahtar, deger) VALUES (?, ?, ?)",
                (self.run_id, anahtar, str(deger))
            )

    def sonlandir(self, durum: str = "FINISHED") -> None:
        bitis_zamani = time.time()
        with self.baglanti:
            self.baglanti.execute(
                "UPDATE kosular SET durum = ?, bitis_zamani = ? WHERE run_id = ?",
                (durum, bitis_zamani, self.run_id)
            )


class MerkeziDeneyTakipMotoru:
    """
    MLflow ve W&B standartlarında merkezi deney kayıt ve yönetim motoru.
    """
    def __init__(self, depo_dizini: str = ".deney_deposu"):
        self.depo_dizini = os.path.abspath(depo_dizini)
        os.makedirs(self.depo_dizini, exist_ok=True)
        self.db_yolu = os.path.join(self.depo_dizini, "deneyler.db")
        self.baglanti = sqlite3.connect(self.db_yolu, check_same_thread=False)
        self._tablolari_olustur()
        self.aktif_kosu: Optional[DeneyKosusu] = None

    def _tablolari_olustur(self) -> None:
        with self.baglanti:
            self.baglanti.executescript("""
                CREATE TABLE IF NOT EXISTS deneyler (
                    experiment_id TEXT PRIMARY KEY,
                    isim TEXT UNIQUE NOT NULL,
                    olusturma_zamani REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kosular (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    durum TEXT NOT NULL,
                    baslangic_zamani REAL NOT NULL,
                    bitis_zamani REAL,
                    FOREIGN KEY(experiment_id) REFERENCES deneyler(experiment_id)
                );

                CREATE TABLE IF NOT EXISTS parametreler (
                    run_id TEXT NOT NULL,
                    anahtar TEXT NOT NULL,
                    deger TEXT NOT NULL,
                    PRIMARY KEY(run_id, anahtar),
                    FOREIGN KEY(run_id) REFERENCES kosular(run_id)
                );

                CREATE TABLE IF NOT EXISTS metrikler (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    anahtar TEXT NOT NULL,
                    deger REAL NOT NULL,
                    step INTEGER,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES kosular(run_id)
                );

                CREATE TABLE IF NOT EXISTS etiketler (
                    run_id TEXT NOT NULL,
                    anahtar TEXT NOT NULL,
                    deger TEXT NOT NULL,
                    PRIMARY KEY(run_id, anahtar),
                    FOREIGN KEY(run_id) REFERENCES kosular(run_id)
                );

                CREATE TABLE IF NOT EXISTS artefaktlar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    dosya_adi TEXT NOT NULL,
                    dosya_yolu TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES kosular(run_id)
                );
            """)

    def deney_olustur_veya_getir(self, deney_adi: str) -> str:
        imlec = self.baglanti.cursor()
        imlec.execute("SELECT experiment_id FROM deneyler WHERE isim = ?", (deney_adi,))
        satir = imlec.fetchone()
        if satir:
            return satir[0]

        exp_id = str(uuid.uuid4())[:8]
        with self.baglanti:
            self.baglanti.execute(
                "INSERT INTO deneyler (experiment_id, isim, olusturma_zamani) VALUES (?, ?, ?)",
                (exp_id, deney_adi, time.time())
            )
        return exp_id

    def start_run(self, deney_adi: str = "Varsayilan_Deney", kosu_adi: Optional[str] = None) -> DeneyKosusu:
        exp_id = self.deney_olustur_veya_getir(deney_adi)
        run_id = f"run_{str(uuid.uuid4())[:8]}"
        baslangic = time.time()

        with self.baglanti:
            self.baglanti.execute(
                "INSERT INTO kosular (run_id, experiment_id, durum, baslangic_zamani) VALUES (?, ?, ?, ?)",
                (run_id, exp_id, "RUNNING", baslangic)
            )

        kosu = DeneyKosusu(
            run_id=run_id,
            experiment_id=exp_id,
            experiment_name=deney_adi,
            depo_dizini=self.depo_dizini,
            baglanti=self.baglanti
        )

        if kosu_adi:
            kosu.set_tag("run_name", kosu_adi)
        kosu.set_tag("cihaz", "cuda" if torch.cuda.is_available() else "cpu")

        self.aktif_kosu = kosu
        return kosu

    def end_run(self, durum: str = "FINISHED") -> None:
        if self.aktif_kosu:
            self.aktif_kosu.sonlandir(durum=durum)
            self.aktif_kosu = None

    def tum_kosulari_getir(self, deney_adi: str) -> List[Dict[str, Any]]:
        exp_id = self.deney_olustur_veya_getir(deney_adi)
        imlec = self.baglanti.cursor()
        imlec.execute("SELECT run_id, durum, baslangic_zamani, bitis_zamani FROM kosular WHERE experiment_id = ?", (exp_id,))
        satirlar = imlec.fetchall()

        sonuclar = []
        for r_id, durum, b_zam, e_zam in satirlar:
            # Parametreler
            imlec.execute("SELECT anahtar, deger FROM parametreler WHERE run_id = ?", (r_id,))
            params = {k: v for k, v in imlec.fetchall()}

            # Etiketler
            imlec.execute("SELECT anahtar, deger FROM etiketler WHERE run_id = ?", (r_id,))
            tags = {k: v for k, v in imlec.fetchall()}

            # En son metrik değerleri
            imlec.execute("""
                SELECT m1.anahtar, m1.deger 
                FROM metrikler m1
                INNER JOIN (
                    SELECT run_id, anahtar, MAX(step) as max_step
                    FROM metrikler
                    WHERE run_id = ?
                    GROUP BY anahtar
                ) m2 ON m1.run_id = m2.run_id AND m1.anahtar = m2.anahtar AND m1.step = m2.max_step
            """, (r_id,))
            metrics = {k: v for k, v in imlec.fetchall()}

            # Metrik zaman serisi geçmişi
            imlec.execute("SELECT anahtar, step, deger, timestamp FROM metrikler WHERE run_id = ? ORDER BY step ASC", (r_id,))
            metric_history: Dict[str, List[Dict[str, Any]]] = {}
            for k, st, val, ts in imlec.fetchall():
                if k not in metric_history:
                    metric_history[k] = []
                metric_history[k].append({"step": st, "value": val, "timestamp": ts})

            sonuclar.append({
                "run_id": r_id,
                "durum": durum,
                "baslangic_zamani": b_zam,
                "bitis_zamani": e_zam,
                "params": params,
                "tags": tags,
                "metrics": metrics,
                "metric_history": metric_history
            })

        return sonuclar
