"""
PyTest Birim Testleri - Day 245: ROS2 Python Entegrasyon Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ros2_motoru import (
    ROS2Message,
    ROS2Node,
    ROS2Executor,
    RobotSensorActuatorPipeline,
)
from src.ros2_profilleyici import ROS2Profilleyici
from src.gorsellestirici import ROS2Gorsellestirici


def test_ros2_node_initialization():
    """1. ROS2Node düğüm adını doğru başlatmalıdır."""
    node = ROS2Node("test_node")
    assert node.node_name == "test_node"
    assert len(node.publishers) == 0


def test_ros2_node_create_publisher_subscriber():
    """2. create_publisher ve create_subscription sözlükleri güncellemelidir."""
    node = ROS2Node("sensor_node")
    node.create_publisher("/data", qos="Reliable")
    node.create_subscription("/cmd", lambda m: None, qos="SensorData")
    assert "/data" in node.publishers
    assert "/cmd" in node.subscriptions


def test_ros2_executor_add_node():
    """3. ROS2Executor düğümleri ve abonelikleri kaydetmelidir."""
    executor = ROS2Executor()
    node = ROS2Node("worker")
    node.create_subscription("/task", lambda m: None)
    executor.add_node(node)
    assert "worker" in executor.dugumler
    assert "/task" in executor.konu_aboneleri


def test_ros2_topic_publish_and_receive():
    """4. publish_message abonenin geri çağırım fonksiyonunu tetiklemelidir."""
    executor = ROS2Executor()
    alindi = []

    sub_node = ROS2Node("sub_node")
    sub_node.create_subscription("/ping", lambda m: alindi.append(m.payload))
    executor.add_node(sub_node)

    pub_node = ROS2Node("pub_node")
    pub_node.create_publisher("/ping")
    executor.add_node(pub_node)

    iletilen = executor.publish_message("pub_node", "/ping", {"val": 42})
    assert iletilen == 1
    assert len(alindi) == 1
    assert alindi[0]["val"] == 42


def test_ros2_service_call_success():
    """5. call_service kayıtlı servise istek yapıp yanıt almalıdır."""
    executor = ROS2Executor()
    srv_node = ROS2Node("srv_node")
    srv_node.create_service("/add", lambda req: {"sum": req["a"] + req["b"]})
    executor.add_node(srv_node)

    resp = executor.call_service("/add", {"a": 10, "b": 20})
    assert resp["sum"] == 30


def test_ros2_service_call_not_found():
    """6. Tanımlı olmayan servis çağrıldığında NOT_FOUND dönmelidir."""
    executor = ROS2Executor()
    resp = executor.call_service("/unknown", {})
    assert resp["durum"] == "NOT_FOUND"


def test_robot_pipeline_simulation():
    """7. RobotSensorActuatorPipeline sensörden eyleyiciye tam akışı işletmelidir."""
    pipeline = RobotSensorActuatorPipeline()
    pipeline.simule_et(kare_sayisi=3)
    assert len(pipeline.alinan_eylemler) == 3
    assert len(pipeline.alinan_eylemler[0]["delta_joints"]) == 7


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. ROS2Gorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_ros2_paneli.png")
    profil = ROS2Profilleyici.basarim_profili_cikar()

    ROS2Gorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
