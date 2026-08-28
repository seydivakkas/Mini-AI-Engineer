"""
Model Kayıt Sistemi, Sürümleme ve Yaşam Döngüsü Motoru (Model Registry)
-----------------------------------------------------------------------
MLflow Model Registry standartlarında Model Sürümleme (v1, v2, v3...),
Aşama Geçişleri (None -> Staging -> Production -> Archived),
Kalite Kapıları (Quality Gates) ve Sıfır Kesintili Geri Alma (Rollback) Motoru.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import time
import json
import shutil
import sqlite3
import torch


class ModelKayitMotoru:
    """
    Üretim modellerini versiyonlayan, aşamalarını yöneten ve denetleyen merkezi kayıt motoru.
    """
    ASAMALAR = ["NONE", "STAGING", "PRODUCTION", "ARCHIVED"]

    def __init__(self, depo_dizini: str = ".model_registry"):
        self.depo_dizini = os.path.abspath(depo_dizini)
        os.makedirs(self.depo_dizini, exist_ok=True)
        self.db_yolu = os.path.join(self.depo_dizini, "registry.db")
        self.baglanti = sqlite3.connect(self.db_yolu, check_same_thread=False)
        self._tablolari_olustur()

    def _tablolari_olustur(self) -> None:
        with self.baglanti:
            self.baglanti.executescript("""
                CREATE TABLE IF NOT EXISTS kayitli_modeller (
                    model_adi TEXT PRIMARY KEY,
                    aciklama TEXT,
                    olusturma_zamani REAL NOT NULL,
                    guncelleme_zamani REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_surumleri (
                    model_adi TEXT NOT NULL,
                    surum_no INTEGER NOT NULL,
                    asama TEXT NOT NULL,
                    run_id TEXT,
                    kaynak_yol TEXT NOT NULL,
                    depolanan_yol TEXT NOT NULL,
                    sema_json TEXT,
                    metrikler_json TEXT,
                    etiketler_json TEXT,
                    olusturma_zamani REAL NOT NULL,
                    PRIMARY KEY(model_adi, surum_no),
                    FOREIGN KEY(model_adi) REFERENCES kayitli_modeller(model_adi)
                );

                CREATE TABLE IF NOT EXISTS asama_gecisleri (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_adi TEXT NOT NULL,
                    surum_no INTEGER NOT NULL,
                    eski_asama TEXT NOT NULL,
                    yeni_asama TEXT NOT NULL,
                    aciklama TEXT,
                    zaman REAL NOT NULL
                );
            """)

    def model_olustur_veya_getir(self, model_adi: str, aciklama: str = "") -> None:
        simdi = time.time()
        with self.baglanti:
            self.baglanti.execute(
                "INSERT OR IGNORE INTO kayitli_modeller (model_adi, aciklama, olusturma_zamani, guncelleme_zamani) VALUES (?, ?, ?, ?)",
                (model_adi, aciklama, simdi, simdi)
            )

    def surum_ekle(
        self,
        model_adi: str,
        kaynak_agirlik_yolu: str,
        run_id: str = "run_manual",
        metrikler: Optional[Dict[str, float]] = None,
        sema: Optional[Dict[str, Any]] = None,
        etiketler: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Kayıtlı modele yeni bir artan sürüm (v1 -> v2 -> v3) ekler ve ağırlıkları kalıcı olarak depolar.
        """
        self.model_olustur_veya_getir(model_adi)

        imlec = self.baglanti.cursor()
        imlec.execute("SELECT COALESCE(MAX(surum_no), 0) + 1 FROM model_surumleri WHERE model_adi = ?", (model_adi,))
        yeni_surum_no = int(imlec.fetchone()[0])

        # Ağırlık dosyasını registry deposuna kopyala
        hedef_klasor = os.path.join(self.depo_dizini, "modeller", model_adi, f"v{yeni_surum_no}")
        os.makedirs(hedef_klasor, exist_ok=True)
        hedef_dosya_yolu = os.path.join(hedef_klasor, os.path.basename(kaynak_agirlik_yolu))

        if os.path.abspath(kaynak_agirlik_yolu) != os.path.abspath(hedef_dosya_yolu):
            shutil.copy2(kaynak_agirlik_yolu, hedef_dosya_yolu)

        simdi = time.time()
        with self.baglanti:
            self.baglanti.execute("""
                INSERT INTO model_surumleri (
                    model_adi, surum_no, asama, run_id, kaynak_yol, depolanan_yol,
                    sema_json, metrikler_json, etiketler_json, olusturma_zamani
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_adi,
                yeni_surum_no,
                "NONE",
                run_id,
                kaynak_agirlik_yolu,
                hedef_dosya_yolu,
                json.dumps(sema or {}),
                json.dumps(metrikler or {}),
                json.dumps(etiketler or {}),
                simdi
            ))

            self.baglanti.execute(
                "UPDATE kayitli_modeller SET guncelleme_zamani = ? WHERE model_adi = ?",
                (simdi, model_adi)
            )

        return yeni_surum_no

    def asama_degistir(
        self,
        model_adi: str,
        surum_no: int,
        yeni_asama: str,
        mevcut_uretimi_arsivle: bool = True,
        aciklama: str = ""
    ) -> None:
        """
        Bir model sürümünün aşamasını değiştirir (ör: STAGING -> PRODUCTION).
        Eğer yeni aşama PRODUCTION ise ve mevcut bir üretim modeli varsa onu otomatik ARCHIVED yapar.
        """
        yeni_asama = yeni_asama.upper()
        assert yeni_asama in self.ASAMALAR, f"Geçersiz aşama: {yeni_asama}. Geçerli aşamalar: {self.ASAMALAR}"

        imlec = self.baglanti.cursor()
        imlec.execute("SELECT asama FROM model_surumleri WHERE model_adi = ? AND surum_no = ?", (model_adi, surum_no))
        satir = imlec.fetchone()
        if not satir:
            raise ValueError(f"{model_adi} modelinde v{surum_no} sürümü bulunamadı!")

        eski_asama = satir[0]
        if eski_asama == yeni_asama:
            return

        simdi = time.time()

        with self.baglanti:
            # Eğer PRODUCTION yapılıyorsa ve mevcut üretim modeli varsa arşivle
            if yeni_asama == "PRODUCTION" and mevcut_uretimi_arsivle:
                imlec.execute(
                    "SELECT surum_no FROM model_surumleri WHERE model_adi = ? AND asama = 'PRODUCTION'",
                    (model_adi,)
                )
                mevcut_prod_surumler = imlec.fetchall()
                for (p_surum,) in mevcut_prod_surumler:
                    self.baglanti.execute(
                        "UPDATE model_surumleri SET asama = 'ARCHIVED' WHERE model_adi = ? AND surum_no = ?",
                        (model_adi, p_surum)
                    )
                    self.baglanti.execute(
                        "INSERT INTO asama_gecisleri (model_adi, surum_no, eski_asama, yeni_asama, aciklama, zaman) VALUES (?, ?, ?, ?, ?, ?)",
                        (model_adi, p_surum, "PRODUCTION", "ARCHIVED", "Yeni üretim sürümü terfi ettiği için arşivlendi", simdi)
                    )

            # Hedef sürümün aşamasını güncelle
            self.baglanti.execute(
                "UPDATE model_surumleri SET asama = ? WHERE model_adi = ? AND surum_no = ?",
                (yeni_asama, model_adi, surum_no)
            )

            # Geçiş günlüğüne kaydet
            self.baglanti.execute(
                "INSERT INTO asama_gecisleri (model_adi, surum_no, eski_asama, yeni_asama, aciklama, zaman) VALUES (?, ?, ?, ?, ?, ?)",
                (model_adi, surum_no, eski_asama, yeni_asama, aciklama, simdi)
            )

            self.baglanti.execute(
                "UPDATE kayitli_modeller SET guncelleme_zamani = ? WHERE model_adi = ?",
                (simdi, model_adi)
            )

    def uretim_modelini_getir(self, model_adi: str) -> Optional[Dict[str, Any]]:
        imlec = self.baglanti.cursor()
        imlec.execute("""
            SELECT model_adi, surum_no, asama, run_id, depolanan_yol, sema_json, metrikler_json, olusturma_zamani
            FROM model_surumleri
            WHERE model_adi = ? AND asama = 'PRODUCTION'
            ORDER BY surum_no DESC LIMIT 1
        """, (model_adi,))
        satir = imlec.fetchone()
        if not satir:
            return None

        return {
            "model_adi": satir[0],
            "surum_no": satir[1],
            "asama": satir[2],
            "run_id": satir[3],
            "depolanan_yol": satir[4],
            "sema": json.loads(satir[5] or "{}"),
            "metrikler": json.loads(satir[6] or "{}"),
            "olusturma_zamani": satir[7]
        }

    def geri_al(self, model_adi: str, aciklama: str = "Acil Geri Alma (Rollback)") -> Dict[str, Any]:
        """
        Üretimdeki hatalı modeli arşive alıp en son ARCHIVED olan stabil modeli tekrar PRODUCTION yapar.
        """
        imlec = self.baglanti.cursor()
        # En son arşivlenen modeli bul
        imlec.execute("""
            SELECT surum_no FROM model_surumleri
            WHERE model_adi = ? AND asama = 'ARCHIVED'
            ORDER BY surum_no DESC LIMIT 1
        """, (model_adi,))
        onceki = imlec.fetchone()

        if not onceki:
            raise RuntimeError(f"{model_adi} için geri alınabilecek (ARCHIVED) bir model sürümü bulunamadı!")

        onceki_surum = int(onceki[0])
        self.asama_degistir(
            model_adi=model_adi,
            surum_no=onceki_surum,
            yeni_asama="PRODUCTION",
            mevcut_uretimi_arsivle=True,
            aciklama=aciklama
        )

        return self.uretim_modelini_getir(model_adi)

    def tum_surumleri_listele(self, model_adi: str) -> List[Dict[str, Any]]:
        imlec = self.baglanti.cursor()
        imlec.execute("""
            SELECT surum_no, asama, run_id, depolanan_yol, metrikler_json, olusturma_zamani
            FROM model_surumleri
            WHERE model_adi = ?
            ORDER BY surum_no ASC
        """, (model_adi,))
        satirlar = imlec.fetchall()

        return [
            {
                "surum_no": s[0],
                "asama": s[1],
                "run_id": s[2],
                "depolanan_yol": s[3],
                "metrikler": json.loads(s[4] or "{}"),
                "olusturma_zamani": s[5]
            }
            for s in satirlar
        ]
