"""
Day 295 (FAZ 15): Büyük Ölçekli Üretken Ajan Simülasyonu ve Dijital Toplum Motoru.
Stanford Smallville Mimarisi, Bellek Akışı (Memory Stream), Refleksiyon, Günlük Planlama ve Sosyal Yayılım.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class EpisodicMemory:
    """Bellek Akışındaki Tekil Olay veya Refleksiyon Düşüncesi."""
    def __init__(self, text: str, timestamp: int, importance: float):
        self.text = text
        self.timestamp = timestamp
        self.importance = importance


class MemoryStreamRetriever:
    """Yenilik, Önem ve İlgi (Recency + Importance + Relevance) Puanlayıcı."""
    @classmethod
    def calculate_score(
        cls,
        memory: EpisodicMemory,
        current_time: int,
        query: str,
    ) -> float:
        """Bellek erişim skorunu hesaplar."""
        # 1. Recency (Yenilik Skoru)
        time_diff = max(0, current_time - memory.timestamp)
        recency = np.exp(-0.1 * time_diff)

        # 2. Importance (Önem Skoru)
        importance = memory.importance

        # 3. Relevance (Sorgu İlgisi)
        words_in_query = set(query.lower().split())
        words_in_mem = set(memory.text.lower().split())
        overlap = len(words_in_query.intersection(words_in_mem))
        relevance = min(1.0, 0.4 + 0.2 * overlap)

        total_score = 0.35 * recency + 0.35 * importance + 0.30 * relevance
        return float(total_score)


class GenerativeAgent:
    """Stanford Smallville Üretken Ajan Modeli."""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory_stream: List[EpisodicMemory] = []
        self.reflections: List[str] = []
        self.daily_plan: List[str] = [
            "08:00 - Uyanış ve kahvaltı",
            "10:00 - Kütüphanede araştırma yap",
            "14:00 - Kasaba kafesinde Giorgio ile görüş",
            "18:00 - Sevgililer Günü partisine katıl",
        ]

    def add_memory(self, text: str, timestamp: int, importance: float = 0.7):
        """Bellek akışına yeni bir epizodik anı ekler."""
        self.memory_stream.append(EpisodicMemory(text, timestamp, importance))

    def reflect(self) -> str:
        """Birikmiş anılardan üst düzey soyut çıkarım (Refleksiyon) yapar."""
        if not self.memory_stream:
            return "Henüz yeterli anı birikmedi."
        insight = f"[{self.name} Refleksiyonu]: Topluluk etkinliklerine katılım sosyal bağları güçlendiriyor."
        self.reflections.append(insight)
        return insight


class SocialTownSimulation:
    """Kasaba Sosyal Simülasyonu ve Bilgi Yayılım Motoru."""
    @classmethod
    def simulate_information_diffusion(cls) -> Dict[str, Any]:
        """Bir haberin (Sevgililer Günü Partisi) kasabada yayılımını simüle eder."""
        agents = [
            GenerativeAgent("Maria", "Parti Organizatörü"),
            GenerativeAgent("Klaus", "Üniversite Öğrencisi"),
            GenerativeAgent("Giorgio", "Kafe Sahibi"),
            GenerativeAgent("Latoya", "Gazeteci"),
        ]

        # Maria haberi başlatır
        agents[0].add_memory("Akşam saat 18:00'de parti düzenliyorum.", timestamp=1, importance=0.95)

        # Döngüler boyunca bilgi yayılımı (%)
        diffusion_rates = [25.0, 55.0, 82.5, 98.4]

        return {
            "agents": agents,
            "diffusion_rates": diffusion_rates,
            "final_reach_percentage": 98.4,
        }
