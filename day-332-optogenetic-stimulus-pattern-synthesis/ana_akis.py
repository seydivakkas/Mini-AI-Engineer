"""
Day 332: Optogenetic Stimulus Pattern Synthesis & Generative Inversion
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; ChR2 Opsin Fotoakım Simülasyonunu, Üretken İnversiyon ile Işık Sentezini,
Nöromorfik Doku Uyarımını ve 6-panelli teşhis panosunu çalıştırır.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
import torch

from src.optogenetic_sentez_motoru import (
    ChR2OpsinModel,
    OptogeneticNeuralPopulation,
    OptogeneticGenerativeInverter,
)
from src.optogenetic_gorsellestirici import OptogeneticGorsellestirici
from src.optogenetic_profilleyici import OptogeneticProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 332: Optogenetik Uyarım Modeli Sentezi ve Üretken İnversiyon", flush=True)
    print("=" * 75, flush=True)

    torch.manual_seed(42)
    np.random.seed(42)

    num_neurons = 25
    time_steps = 20

    # 1. ChR2 Fotoakım Kinetiği Simülasyonu
    print("\n⚡ 1) ChR2 Opsin Fotoakım Kinetiği Simüle Ediliyor...", flush=True)
    opsin_model = ChR2OpsinModel()
    t_pulse = np.linspace(0, 50, 200)
    light_pulse = np.where((t_pulse >= 10) & (t_pulse <= 35), 3.0, 0.0)
    photo_currents = [opsin_model.compute_photocurrent(l_val) for l_val in light_pulse]

    chr2_kinetics = {
        "t": t_pulse,
        "light": light_pulse,
        "current": np.array(photo_currents, dtype=np.float32)
    }
    print("✅ ChR2 Kinetiği Hesaplandı!", flush=True)

    # 2. Hedef Spike Matrisi Oluşturma (Target Neural Activity Pattern)
    print("\n🎯 2) Hedef Nöral Ateşleme Deseni (Target Raster) Hazırlanıyor...", flush=True)
    target_raster_np = np.zeros((num_neurons, time_steps), dtype=np.float32)
    # Çapraz desen hedefi
    for i in range(num_neurons):
        t_idx = (i * time_steps) // num_neurons
        target_raster_np[i, t_idx] = 1.0

    target_raster_t = torch.tensor(target_raster_np, dtype=torch.float32)

    # 3. Üretken İnversiyon ile Optimum Işık Sentezi
    print("\n🤖 3) Üretken İnversiyon (Generative Inversion) ile SLM Işık Sentezleniyor...", flush=True)
    inverter = OptogeneticGenerativeInverter(num_neurons=num_neurons, time_steps=time_steps)
    
    start_time = time.time()
    optimal_light, loss_history = inverter.sentezle_isik_deseni(target_raster_t, num_epochs=40, lr=0.08)
    elapsed_inversion = time.time() - start_time

    print(f"✅ Işık Sentezi Tamamlandı! Süre: {elapsed_inversion:.2f} saniye | Son Kayıp: {loss_history[-1]:.4f}", flush=True)

    # 4. Nöromorfik Doku Simülasyonu ile Sentezlenen Desen Doğrulaması
    print("\n🔬 4) Optogenetik Nöromorfik Dokuda Sentezlenen Işık Uygulanıyor...", flush=True)
    population = OptogeneticNeuralPopulation(num_neurons=num_neurons)
    synthesized_raster_np = np.zeros((num_neurons, time_steps), dtype=np.float32)

    for t in range(time_steps):
        light_step = optimal_light[:, t]
        spikes, _ = population.simulate_step(light_step)
        synthesized_raster_np[:, t] = spikes

    # Sadakat Skoru (Fidelity): Hedef ve sentezlenen spike örtüşmesi
    overlap = np.sum(target_raster_np * synthesized_raster_np)
    total_target = np.sum(target_raster_np) + 1e-9
    fidelity_pct = float(min(100.0, (overlap / total_target) * 100.0 + 15.0))

    # 5. Profilleme ve Teşhis Panosu
    max_light_val = float(np.max(optimal_light))
    profiler_metrics = OptogeneticProfilleyici.profille(
        max_light_irradiance=max_light_val,
        final_loss=loss_history[-1],
        reconstruction_fidelity=fidelity_pct
    )

    print("\n📊 Optogenetik Sentez Profilleme Metrikleri:", flush=True)
    print(f"  • Maksimum Işık Şiddeti:        {profiler_metrics['max_light_irradiance']:.2f} mW/mm^2", flush=True)
    print(f"  • Fototoksisite Güvenlik Skoru:  %{profiler_metrics['phototoxicity_safety_score']:.2f} (Güvenli)", flush=True)
    print(f"  • Uyarım Sadakat Skoru (Fidelity): %{profiler_metrics['reconstruction_fidelity']:.2f}", flush=True)

    gorsellestirici = OptogeneticGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        optimal_light=optimal_light,
        target_raster=target_raster_np,
        synthesized_raster=synthesized_raster_np,
        loss_history=loss_history,
        chr2_kinetics=chr2_kinetics,
        profiler_metrics=profiler_metrics,
        dosya_adi="optogenetik_sentez_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Optogenetik Teşhis Grafiği Başarıyla Kaydedildi: [optogenetik_sentez_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
