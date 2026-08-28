"""
Embedding Geometrisi, Kosinüs Benzerliği, İzotropi ve Linear Probing Analizörü.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class EmbeddingGeometriAnalizoru:
    """Temsil uzayının geometrisini, sınıf-içi/sınıf-dışı kosinüs benzerliğini ve ayrışabilirliğini ölçer."""

    @staticmethod
    def l2_norm_dogrula(embeddings: np.ndarray) -> Dict[str, float]:
        """Tüm vektörlerin birim hiperküre (Unit Sphere) üzerinde olup olmadığını denetler."""
        normlar = np.linalg.norm(embeddings, axis=1)
        return {
            "ort_norm": float(np.mean(normlar)),
            "min_norm": float(np.min(normlar)),
            "maks_norm": float(np.max(normlar)),
            "norm_std": float(np.std(normlar)),
            "birim_kure_gecerli_mi": bool(np.allclose(normlar, 1.0, atol=1e-4))
        }

    @staticmethod
    def benzerlik_analizi(
        embeddings: np.ndarray,
        labels: np.ndarray,
        maks_ornek: int = 1000
    ) -> Dict[str, Any]:
        """İkili kosinüs benzerliği matrisini hesaplar ve sınıf-içi vs sınıf-dışı benzerlik dağılımını çıkarır."""
        n = min(len(embeddings), maks_ornek)
        alt_emb = embeddings[:n]
        alt_labels = labels[:n]

        # Vektörler L2 normalize olduğundan Kosinüs Benzerliği = Matris Çarpımıdır!
        benzerlik_matrisi = np.dot(alt_emb, alt_emb.T)

        siniflar = np.unique(alt_labels)
        num_classes = len(siniflar)
        sinif_ortalama_matrisi = np.zeros((num_classes, num_classes), dtype=np.float32)

        intra_benzerlikler: List[float] = []
        inter_benzerlikler: List[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                benzerlik = float(benzerlik_matrisi[i, j])
                if alt_labels[i] == alt_labels[j]:
                    intra_benzerlikler.append(benzerlik)
                else:
                    inter_benzerlikler.append(benzerlik)

        for c1_idx, c1 in enumerate(siniflar):
            for c2_idx, c2 in enumerate(siniflar):
                mask1 = (alt_labels == c1)
                mask2 = (alt_labels == c2)
                alt_sim = benzerlik_matrisi[np.ix_(mask1, mask2)]
                sinif_ortalama_matrisi[c1_idx, c2_idx] = float(np.mean(alt_sim))

        ort_intra = float(np.mean(intra_benzerlikler)) if intra_benzerlikler else 0.0
        ort_inter = float(np.mean(inter_benzerlikler)) if inter_benzerlikler else 0.0
        if ort_inter > 0:
            ayrisabilirlik = float(ort_intra / ort_inter)
        else:
            ayrisabilirlik = float(ort_intra / 1e-6) if ort_intra > 0 else 1.0

        return {
            "benzerlik_matrisi": benzerlik_matrisi,
            "sinif_ortalama_matrisi": sinif_ortalama_matrisi,
            "siniflar": siniflar.tolist(),
            "intra_benzerlikler": intra_benzerlikler,
            "inter_benzerlikler": inter_benzerlikler,
            "ort_intra_benzerlik": ort_intra,
            "ort_inter_benzerlik": ort_inter,
            "ayrisabilirlik_orani": ayrisabilirlik
        }

    @staticmethod
    def svd_ve_izotropi_analizi(embeddings: np.ndarray) -> Dict[str, Any]:
        """Temsil uzayının tekil değer spektrumunu (SVD) ve izotropi/boyutluluk kapasitesini analiz eder."""
        merkezlenmis = embeddings - np.mean(embeddings, axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(merkezlenmis, full_matrices=False)

        toplam_enerji = np.sum(s ** 2)
        varyans_oranlari = (s ** 2) / max(toplam_enerji, 1e-12)
        kumulatif_varyans = np.cumsum(varyans_oranlari)

        # Efektif Boyut (Participation Ratio)
        efektif_boyut = float((np.sum(s ** 2) ** 2) / np.sum(s ** 4))
        izotropi_skoru = float(np.min(s) / max(np.max(s), 1e-12))

        return {
            "tekil_degerler": s[:30].tolist(),
            "varyans_oranlari": varyans_oranlari[:30].tolist(),
            "kumulatif_varyans": kumulatif_varyans[:30].tolist(),
            "efektif_boyut": efektif_boyut,
            "izotropi_skoru": izotropi_skoru
        }

    @staticmethod
    def linear_probing_egit(
        train_emb: np.ndarray,
        train_y: np.ndarray,
        val_emb: np.ndarray,
        val_y: np.ndarray,
        num_classes: int = 10,
        epochs: int = 15,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """Dondurulmuş özellikler üzerine doğrusal sınıflandırma başlığı (Linear Probe) eğitir."""
        torch.manual_seed(42)
        embed_dim = train_emb.shape[1]

        probe = nn.Linear(embed_dim, num_classes)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_emb).float(), torch.from_numpy(train_y).long()),
            batch_size=32, shuffle=True
        )

        val_x = torch.from_numpy(val_emb).float()
        val_targets = torch.from_numpy(val_y).long()

        dogruluk_gecmisi: List[float] = []

        for epoch in range(1, epochs + 1):
            probe.train()
            for bx, by in train_loader:
                optimizer.zero_grad(set_to_none=True)
                out = probe(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            probe.eval()
            with torch.no_grad():
                val_out = probe(val_x)
                preds = val_out.argmax(dim=-1)
                acc = float((preds == val_targets).sum().item() / len(val_targets) * 100.0)
                dogruluk_gecmisi.append(acc)

        return {
            "nihai_val_dogruluk": dogruluk_gecmisi[-1],
            "dogruluk_gecmisi": dogruluk_gecmisi
        }
