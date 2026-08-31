/**
 * Tesla CV-NF Real-Time 4D Driving Simulator & Gamified Testing Arena
 * Fully Functional 8-Camera HW4 Surround Vision Engine,
 * Interactive Camera Inspector Modal, 4D Semantic Layers,
 * Target Lock (<kbd>G</kbd>) & FSD Auto-Evasion (<kbd>E</kbd>)
 *
 * Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
 * All Rights Reserved.
 */

// 1. Simulator Configurations
const SIM_CONFIG = {
    MODES: {
        STANDARD: 'STANDARD_30FPS',
        CVNF: 'TESLA_CVNF_1000HZ'
    },
    LAYERS: {
        NORMAL: 'NORMAL',
        SALIENCY: 'SALIENCY',
        SEMANTIC: 'SEMANTIC',
        UNCERTAINTY: 'UNCERTAINTY'
    },
    WEATHER: {
        CLEAR: 'CLEAR',
        FOG: 'FOG',
        RAIN: 'RAIN'
    },
    MAX_SPEED_KMH: 140,
    ACCEL: 1.2,
    BRAKE: 2.5,
    FRICTION: 0.992,
    MAX_STEER_ANGLE: 0.18,
    LANE_SLIDE_SPEED: 7.5,
    TARGET_FOLLOW_DIST: 18.0
};

let currentMode = SIM_CONFIG.MODES.CVNF;
let currentLayer = SIM_CONFIG.LAYERS.NORMAL;
let currentWeather = SIM_CONFIG.WEATHER.CLEAR;
let currentMission = 1;
let cameraView = 'CHASE';
let autoEvasionActive = false;
let targetLockActive = false;
let lockedTarget = null;
let audioEnabled = true;

let saliencyAlpha = 0.85;
let uncertaintyThreshold = 0.40;

// Active Modal Camera Inspector
let inspectedCamKey = 'frontWide';
let isCameraModalOpen = false;

// 🔗 1.1 Gerçek PyTorch Backend WebSocket Telemetri Bağlantısı
let telemetrySocket = null;
let isBackendConnected = false;

function connectBackendWebSocket() {
    try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
        telemetrySocket = new WebSocket(wsUrl);

        telemetrySocket.onopen = () => {
            isBackendConnected = true;
            console.log("✅ Gerçek Tesla CV-NF PyTorch NPU Backend'e Bağlanıldı!");
            const badge = document.getElementById('backend-status-badge');
            if (badge) {
                badge.innerText = "🟢 GERÇEK PYTORCH NPU AKTİF";
                badge.style.color = "var(--tesla-green)";
            }
        };

        telemetrySocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                // Canlı PyTorch NPU Gecikmesi ve Sanal Yenileme Hızı
                const elLatency = document.getElementById('val-latency');
                const elHz = document.getElementById('val-fps');
                const elEvents = document.getElementById('modal-cam-events');
                const elEpistemic = document.getElementById('val-epistemic');
                const elAleatoric = document.getElementById('val-aleatoric');
                const elHji = document.getElementById('val-hji-barrier');

                if (elLatency) elLatency.innerText = `${data.hw4_npu_latency_ms} ms`;
                if (elHz) elHz.innerText = `${data.virtual_hz} Hz`;
                if (elEvents) elEvents.innerText = `${data.event_rate_mev_s}M Olay/s (NPU)`;
                
                if (elEpistemic && data.epistemic_ood !== undefined) {
                    elEpistemic.innerText = `${data.epistemic_ood.toFixed(3)} (BİLİNİYOR)`;
                }
                if (elAleatoric && data.aleatoric_noise !== undefined) {
                    elAleatoric.innerText = `${data.aleatoric_noise.toFixed(3)} (SENSÖR)`;
                }
                if (elHji && data.hji_barrier_margin_m !== undefined) {
                    elHji.innerText = `+${data.hji_barrier_margin_m.toFixed(2)} m (KORUMALI)`;
                }
            } catch (err) {}
        };

        telemetrySocket.onclose = () => {
            isBackendConnected = false;
            setTimeout(connectBackendWebSocket, 3000);
        };
    } catch (e) {
        console.warn("WebSocket bağlantısı kurulamadı:", e);
    }
}

connectBackendWebSocket();

// 2. Web Audio API Synthesizer
let audioCtx = null;
function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
}

function playRadarPing(pan = 0, freq = 880, duration = 0.12) {
    if (!audioEnabled || !audioCtx) return;
    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        const panner = audioCtx.createStereoPanner ? audioCtx.createStereoPanner() : null;

        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(freq * 1.5, audioCtx.currentTime + duration);

        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);

        if (panner) {
            panner.pan.setValueAtTime(pan, audioCtx.currentTime);
            osc.connect(gain).connect(panner).connect(audioCtx.destination);
        } else {
            osc.connect(gain).connect(audioCtx.destination);
        }

        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {}
}

function playLockSound(isLocking = true) {
    if (!audioEnabled || !audioCtx) return;
    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'triangle';
        const startFreq = isLocking ? 600 : 1200;
        const endFreq = isLocking ? 1400 : 400;

        osc.frequency.setValueAtTime(startFreq, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(endFreq, audioCtx.currentTime + 0.2);

        gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.2);

        osc.connect(gain).connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.2);
    } catch (e) {}
}

window.toggleAudioSynth = function() {
    audioEnabled = !audioEnabled;
    const btn = document.getElementById('btn-sound-toggle');
    if (btn) {
        btn.innerText = audioEnabled ? "🔊 Sesli NPU Synth: AÇIK" : "🔇 Sesli NPU Synth: KAPALI";
    }
};

// 3. Three.js Scene, Camera, and Renderer Setup
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x08090c);
scene.fog = new THREE.FogExp2(0x08090c, 0.010);

const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.1, 1800);
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
dirLight.position.set(30, 80, 40);
dirLight.castShadow = true;
scene.add(dirLight);

// 4. Build Road, Barriers & Environment
const ROAD_WIDTH = 22;
const ROAD_LENGTH = 1600;

const roadGeo = new THREE.PlaneGeometry(ROAD_WIDTH, ROAD_LENGTH, 10, 100);
const roadMat = new THREE.MeshStandardMaterial({ color: 0x14171d, roughness: 0.75 });
const road = new THREE.Mesh(roadGeo, roadMat);
road.rotation.x = -Math.PI / 2;
road.position.z = -ROAD_LENGTH / 2 + 60;
road.receiveShadow = true;
scene.add(road);

// Barriers
const barrierGeo = new THREE.BoxGeometry(0.5, 0.8, ROAD_LENGTH);
const barrierMat = new THREE.MeshStandardMaterial({ color: 0x30363d });
const leftBarrier = new THREE.Mesh(barrierGeo, barrierMat);
leftBarrier.position.set(-ROAD_WIDTH/2 - 0.25, 0.4, -ROAD_LENGTH / 2 + 60);
const rightBarrier = new THREE.Mesh(barrierGeo, barrierMat);
rightBarrier.position.set(ROAD_WIDTH/2 + 0.25, 0.4, -ROAD_LENGTH / 2 + 60);
scene.add(leftBarrier, rightBarrier);

// Ground Grid
const gridHelper = new THREE.GridHelper(1600, 80, 0x00f0ff, 0x1f242c);
gridHelper.position.set(0, -0.05, -ROAD_LENGTH / 2 + 60);
scene.add(gridHelper);

// Lane Dividers
const laneGroup = new THREE.Group();
const stripeGeo = new THREE.BoxGeometry(0.28, 0.04, 5);
const stripeMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
for (let z = 60; z > -ROAD_LENGTH + 60; z -= 10) {
    const leftStripe = new THREE.Mesh(stripeGeo, stripeMat);
    leftStripe.position.set(-3.8, 0.02, z);
    const rightStripe = new THREE.Mesh(stripeGeo, stripeMat);
    rightStripe.position.set(3.8, 0.02, z);
    laneGroup.add(leftStripe, rightStripe);
}
scene.add(laneGroup);

// Saliency Attention Cone
const saliencyConeGeo = new THREE.ConeGeometry(14, 50, 24, 1, true);
const saliencyConeMat = new THREE.MeshBasicMaterial({ color: 0xff003c, transparent: true, opacity: 0.0, side: THREE.DoubleSide, wireframe: true });
const saliencyCone = new THREE.Mesh(saliencyConeGeo, saliencyConeMat);
saliencyCone.rotation.x = Math.PI / 2;
saliencyCone.position.set(0, 0.5, -25);
scene.add(saliencyCone);

// Tunnel Structure
const tunnelGroup = new THREE.Group();
const tunnelLength = 300;
const tunnelZ = -600;
const tunnelGeo = new THREE.CylinderGeometry(ROAD_WIDTH / 2 + 2, ROAD_WIDTH / 2 + 2, tunnelLength, 24, 1, true, 0, Math.PI);
const tunnelMat = new THREE.MeshStandardMaterial({ color: 0x090b0e, roughness: 0.95, side: THREE.DoubleSide });
const tunnelMesh = new THREE.Mesh(tunnelGeo, tunnelMat);
tunnelMesh.rotation.z = Math.PI / 2;
tunnelMesh.rotation.y = Math.PI / 2;
tunnelMesh.position.set(0, 0, tunnelZ);
tunnelGroup.add(tunnelMesh);

const tunnelLightSources = [];
for (let tz = tunnelZ + tunnelLength/2 - 20; tz > tunnelZ - tunnelLength/2 + 20; tz -= 45) {
    const tLight = new THREE.PointLight(0xffb800, 2.5, 30);
    tLight.position.set(0, 7, tz);
    tunnelGroup.add(tLight);
    tunnelLightSources.push(tLight);
}
scene.add(tunnelGroup);

// 5. Precision Tesla Vehicle Entity
const carGroup = new THREE.Group();

const bodyGeo = new THREE.BoxGeometry(2.4, 0.85, 4.8);
const bodyMat = new THREE.MeshStandardMaterial({ color: 0xd0d5dd, metalness: 0.85, roughness: 0.25 });
const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
bodyMesh.position.y = 0.7;
bodyMesh.castShadow = true;
carGroup.add(bodyMesh);

const roofGeo = new THREE.BoxGeometry(1.9, 0.6, 2.4);
const roofMat = new THREE.MeshStandardMaterial({ color: 0x161b22, roughness: 0.1, metalness: 0.9 });
const roofMesh = new THREE.Mesh(roofGeo, roofMat);
roofMesh.position.set(0, 1.3, -0.3);
carGroup.add(roofMesh);

// 360° Roof LiDAR Pod & Spinner
const lidarPodGeo = new THREE.CylinderGeometry(0.28, 0.32, 0.25, 20);
const lidarPodMat = new THREE.MeshStandardMaterial({ color: 0x1a202c, metalness: 0.9, roughness: 0.2 });
const lidarPod = new THREE.Mesh(lidarPodGeo, lidarPodMat);
lidarPod.position.set(0, 1.72, -0.3);
carGroup.add(lidarPod);

const lidarSpinnerGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.10, 16);
const lidarSpinnerMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
const lidarSpinner = new THREE.Mesh(lidarSpinnerGeo, lidarSpinnerMat);
lidarSpinner.position.set(0, 1.88, -0.3);
carGroup.add(lidarSpinner);

// LiDAR Ring & Blind Spot Arcs
const lidarRingGeo = new THREE.RingGeometry(3.5, 14.0, 36);
const lidarRingMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.18, side: THREE.DoubleSide });
const lidarRing = new THREE.Mesh(lidarRingGeo, lidarRingMat);
lidarRing.rotation.x = -Math.PI / 2;
lidarRing.position.set(0, 0.05, 0);
carGroup.add(lidarRing);

const blindSpotGeo = new THREE.RingGeometry(2.0, 4.5, 18, 1, 0, Math.PI);
const leftBlindMat = new THREE.MeshBasicMaterial({ color: 0xff003c, transparent: true, opacity: 0.0, side: THREE.DoubleSide });
const leftBlindArc = new THREE.Mesh(blindSpotGeo, leftBlindMat);
leftBlindArc.rotation.x = -Math.PI / 2;
leftBlindArc.rotation.z = Math.PI / 2;
leftBlindArc.position.set(-1.8, 0.06, 0);
carGroup.add(leftBlindArc);

const rightBlindMat = new THREE.MeshBasicMaterial({ color: 0xff003c, transparent: true, opacity: 0.0, side: THREE.DoubleSide });
const rightBlindArc = new THREE.Mesh(blindSpotGeo, rightBlindMat);
rightBlindArc.rotation.x = -Math.PI / 2;
rightBlindArc.rotation.z = -Math.PI / 2;
rightBlindArc.position.set(1.8, 0.06, 0);
carGroup.add(rightBlindArc);

// Headlights & Tail Lights
const frontLightGeo = new THREE.BoxGeometry(2.2, 0.08, 0.1);
const frontLightMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
const frontLight = new THREE.Mesh(frontLightGeo, frontLightMat);
frontLight.position.set(0, 0.7, -2.41);
carGroup.add(frontLight);

const rearLightGeo = new THREE.BoxGeometry(2.2, 0.08, 0.1);
const rearLightMat = new THREE.MeshBasicMaterial({ color: 0xe82127 });
const rearLight = new THREE.Mesh(rearLightGeo, rearLightMat);
rearLight.position.set(0, 0.7, 2.41);
carGroup.add(rearLight);

// Wheels
const wheelGeo = new THREE.CylinderGeometry(0.45, 0.45, 0.35, 16);
const wheelMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.6 });

const frontLeftPivot = new THREE.Group();
frontLeftPivot.position.set(-1.25, 0.45, -1.5);
const frontLeftWheel = new THREE.Mesh(wheelGeo, wheelMat);
frontLeftWheel.rotation.z = Math.PI / 2;
frontLeftPivot.add(frontLeftWheel);
carGroup.add(frontLeftPivot);

const frontRightPivot = new THREE.Group();
frontRightPivot.position.set(1.25, 0.45, -1.5);
const frontRightWheel = new THREE.Mesh(wheelGeo, wheelMat);
frontRightWheel.rotation.z = Math.PI / 2;
frontRightPivot.add(frontRightWheel);
carGroup.add(frontRightPivot);

const rearLeftPivot = new THREE.Group();
rearLeftPivot.position.set(-1.25, 0.45, 1.5);
const rearLeftWheel = new THREE.Mesh(wheelGeo, wheelMat);
rearLeftWheel.rotation.z = Math.PI / 2;
rearLeftPivot.add(rearLeftWheel);
carGroup.add(rearLeftPivot);

const rearRightPivot = new THREE.Group();
rearRightPivot.position.set(1.25, 0.45, 1.5);
const rearRightWheel = new THREE.Mesh(wheelGeo, wheelMat);
rearRightWheel.rotation.z = Math.PI / 2;
rearRightPivot.add(rearRightWheel);
carGroup.add(rearRightPivot);

carGroup.position.set(0, 0, 30);
scene.add(carGroup);

// 6. FSD Predictive Trajectory Ribbon
const trajectoryCurve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(0, 0.05, 0),
    new THREE.Vector3(0, 0.05, -25),
    new THREE.Vector3(0, 0.05, -55)
);
const trajectoryGeo = new THREE.TubeGeometry(trajectoryCurve, 20, 0.35, 8, false);
const trajectoryMat = new THREE.MeshBasicMaterial({ color: 0x00ff88, transparent: true, opacity: 0.75 });
const trajectoryMesh = new THREE.Mesh(trajectoryGeo, trajectoryMat);
carGroup.add(trajectoryMesh);

// 🛡️ 6.1 3D Hamilton-Jacobi-Isaacs (HJI) Reachability Safety Corridor Envelope
const hjiTubeCurve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(0, 0.4, 0),
    new THREE.Vector3(0, 0.4, -20),
    new THREE.Vector3(0, 0.4, -40)
);
const hjiTubeGeo = new THREE.TubeGeometry(hjiTubeCurve, 24, 1.85, 16, false);
const hjiTubeMat = new THREE.MeshBasicMaterial({ color: 0x00ff88, transparent: true, opacity: 0.22, wireframe: true, side: THREE.DoubleSide });
const hjiTubeMesh = new THREE.Mesh(hjiTubeGeo, hjiTubeMat);
carGroup.add(hjiTubeMesh);

// 7. 3D Target Lock-On Reticle Mesh
const targetLockGroup = new THREE.Group();
const reticleRingGeo = new THREE.RingGeometry(1.8, 2.2, 32);
const reticleRingMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, side: THREE.DoubleSide, transparent: true, opacity: 0.85 });
const reticleRing = new THREE.Mesh(reticleRingGeo, reticleRingMat);
reticleRing.rotation.x = -Math.PI / 2;
targetLockGroup.add(reticleRing);

const lockBracketGeo = new THREE.BoxGeometry(0.15, 0.8, 0.8);
const lockBracketMat = new THREE.MeshBasicMaterial({ color: 0x00ff88 });
const bracketLeft = new THREE.Mesh(lockBracketGeo, lockBracketMat);
bracketLeft.position.set(-1.8, 0.4, 0);
const bracketRight = new THREE.Mesh(lockBracketGeo, lockBracketMat);
bracketRight.position.set(1.8, 0.4, 0);
targetLockGroup.add(bracketLeft, bracketRight);
targetLockGroup.visible = false;
scene.add(targetLockGroup);

// 8. Track Obstacles & Dynamic Lead Vehicle
const obstacles = [
    { x: -3.8, z: -80, type: 'LEAD_VEHICLE', label: 'LEAD-01', semantic: 'VEHICLE', size: [2.3, 1.1, 4.8], speed: -0.42, color: 0x00f0ff, uncertainty: 0.32, saliency: 0.98, isVehicle: true },
    { x: 0, z: -200, type: 'BARRIER', label: 'BARRIER', semantic: 'HAZARD', size: [6, 1.2, 1], color: 0xff003c, uncertainty: 0.62, saliency: 0.95 },
    { x: 3.8, z: -360, type: 'CUT_IN_CAR', label: 'SEDAN-02', semantic: 'VEHICLE', size: [2.2, 1, 4.5], speed: -0.36, color: 0x00f0ff, uncertainty: 0.55, saliency: 0.88, isVehicle: true },
    { x: 3.8, z: -630, type: 'TUNNEL_HAZARD', label: 'DEBRIS', semantic: 'HAZARD', size: [3.5, 1, 1], color: 0xff9500, uncertainty: 0.78, saliency: 0.92 },
    { x: -2, z: -940, type: 'PEDESTRIAN', label: 'PED-01', semantic: 'PEDESTRIAN', size: [1.2, 1.8, 1.2], color: 0xffe600, uncertainty: 0.48, saliency: 0.75 }
];

const obstacleMeshes = [];
obstacles.forEach(obs => {
    const group = new THREE.Group();
    group.position.set(obs.x, obs.size[1]/2, obs.z);

    const geo = new THREE.BoxGeometry(...obs.size);
    const mat = new THREE.MeshStandardMaterial({ color: obs.color, metalness: 0.6, roughness: 0.4 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    group.add(mesh);

    const saliencyGeo = new THREE.BoxGeometry(obs.size[0] + 0.8, obs.size[1] + 0.8, obs.size[2] + 0.8);
    const saliencyMat = new THREE.MeshBasicMaterial({ color: 0xff003c, transparent: true, opacity: 0.0, wireframe: true });
    const saliencyMesh = new THREE.Mesh(saliencyGeo, saliencyMat);
    group.add(saliencyMesh);

    const dangerGeo = new THREE.SphereGeometry(Math.max(...obs.size) * 0.9, 16, 16);
    const dangerMat = new THREE.MeshBasicMaterial({ color: 0xbd00ff, transparent: true, opacity: 0.0, wireframe: true });
    const dangerDome = new THREE.Mesh(dangerGeo, dangerMat);
    dangerDome.position.y = 0.5;
    group.add(dangerDome);

    const semanticGeo = new THREE.BoxGeometry(obs.size[0] + 0.3, obs.size[1] + 0.3, obs.size[2] + 0.3);
    const semanticMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.0, wireframe: false });
    const semanticShell = new THREE.Mesh(semanticGeo, semanticMat);
    group.add(semanticShell);

    scene.add(group);
    obstacleMeshes.push({ group, mesh, saliencyMesh, dangerDome, semanticShell, data: obs });
});

// Rain System
const rainCount = 1500;
const rainGeo = new THREE.BufferGeometry();
const rainPositions = new Float32Array(rainCount * 3);
for (let i = 0; i < rainCount * 3; i += 3) {
    rainPositions[i] = (Math.random() - 0.5) * 60;
    rainPositions[i + 1] = Math.random() * 30;
    rainPositions[i + 2] = (Math.random() - 0.5) * 100;
}
rainGeo.setAttribute('position', new THREE.BufferAttribute(rainPositions, 3));
const rainMat = new THREE.PointsMaterial({ color: 0x9ac7d9, size: 0.15, transparent: true, opacity: 0.0 });
const rainParticles = new THREE.Points(rainGeo, rainMat);
scene.add(rainParticles);

// 📹 9. High-Precision 8-Camera HW4 Surround Vision Engine
const CAMERA_RIG = {
    frontWide: { angle: 0, fov: 120, focal: '2.1mm', fNumber: 'f/1.7', range: 90, label: 'FRONT WIDE 120°' },
    frontMain: { angle: 0, fov: 50, focal: '5.2mm', fNumber: 'f/1.8', range: 160, label: 'FRONT MAIN 50°' },
    frontNarrow: { angle: 0, fov: 28, focal: '12.0mm', fNumber: 'f/2.0', range: 250, label: 'FRONT NARROW 28°' },
    rear: { angle: Math.PI, fov: 135, focal: '1.9mm', fNumber: 'f/1.6', range: 75, label: 'REAR BACKUP 135°' },
    leftPillar: { angle: -Math.PI / 4, fov: 90, focal: '3.5mm', fNumber: 'f/1.7', range: 110, label: 'LEFT PILLAR 90°' },
    rightPillar: { angle: Math.PI / 4, fov: 90, focal: '3.5mm', fNumber: 'f/1.7', range: 110, label: 'RIGHT PILLAR 90°' },
    leftRepeater: { angle: -3 * Math.PI / 4, fov: 90, focal: '3.5mm', fNumber: 'f/1.7', range: 85, label: 'LEFT REPEATER (BLIND SPOT)' },
    rightRepeater: { angle: 3 * Math.PI / 4, fov: 90, focal: '3.5mm', fNumber: 'f/1.7', range: 85, label: 'RIGHT REPEATER (BLIND SPOT)' }
};

const cameraCanvases = {
    frontWide: document.getElementById('cam-front-wide'),
    frontMain: document.getElementById('cam-front-main'),
    frontNarrow: document.getElementById('cam-front-narrow'),
    rear: document.getElementById('cam-rear'),
    leftPillar: document.getElementById('cam-left-pillar'),
    rightPillar: document.getElementById('cam-right-pillar'),
    leftRepeater: document.getElementById('cam-left-repeater'),
    rightRepeater: document.getElementById('cam-right-repeater')
};

Object.values(cameraCanvases).forEach(c => {
    if (c) { c.width = 160; c.height = 80; }
});

const modalCanvas = document.getElementById('modal-cam-canvas');
if (modalCanvas) { modalCanvas.width = 580; modalCanvas.height = 280; }

window.openCameraModal = function(camKey, title) {
    inspectedCamKey = camKey;
    isCameraModalOpen = true;
    const modal = document.getElementById('camera-modal');
    const modalTitle = document.getElementById('modal-cam-title');
    const modalFocal = document.getElementById('modal-cam-focal');
    if (modal) modal.style.display = 'flex';
    if (modalTitle) modalTitle.innerText = `📹 TESLA HW4: ${title}`;
    if (modalFocal && CAMERA_RIG[camKey]) {
        modalFocal.innerText = `${CAMERA_RIG[camKey].focal} (${CAMERA_RIG[camKey].fNumber}) — FOV: ${CAMERA_RIG[camKey].fov}°`;
    }
};

window.closeCameraModal = function() {
    isCameraModalOpen = false;
    const modal = document.getElementById('camera-modal');
    if (modal) modal.style.display = 'none';
};

// Render Individual Camera Viewport with Genuine 3D Perspective Projection
function renderSingleCameraView(ctx, width, height, camKey, speedKmh, now) {
    const config = CAMERA_RIG[camKey];
    if (!config) return;

    const inTunnel = (carGroup.position.z < -450 && carGroup.position.z > -750);
    const isStandardBlind = (currentMode === SIM_CONFIG.MODES.STANDARD) && inTunnel;

    // 1. Clear Screen / Sky / Tunnel Background
    if (isStandardBlind) {
        // Standard Camera Motion Blur / HDR Whiteout
        ctx.fillStyle = inTunnel ? '#FFFFFF' : '#111';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#E82127';
        ctx.font = 'bold 12px JetBrains Mono';
        ctx.fillText('⚠️ BLIND / HDR SATURATION', width * 0.1, height * 0.5);
        ctx.fillStyle = '#666';
        ctx.font = '9px JetBrains Mono';
        ctx.fillText('Standard 30 FPS Sensor Blind Spot', width * 0.1, height * 0.65);
        return;
    }

    // Sky gradient
    const skyGrad = ctx.createLinearGradient(0, 0, 0, height);
    if (currentMode === SIM_CONFIG.MODES.CVNF) {
        skyGrad.addColorStop(0, '#04070c');
        skyGrad.addColorStop(0.5, '#09131d');
        skyGrad.addColorStop(1, '#051824');
    } else {
        skyGrad.addColorStop(0, '#020305');
        skyGrad.addColorStop(1, '#11151c');
    }
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, width, height);

    // 2. Road Horizon & Ground
    const horizonY = height * 0.48;
    ctx.fillStyle = currentLayer === SIM_CONFIG.LAYERS.SEMANTIC ? '#063319' : (currentLayer === SIM_CONFIG.LAYERS.SALIENCY ? '#0b1120' : '#12161f');
    ctx.fillRect(0, horizonY, width, height - horizonY);

    // 3. Perspective Road Borders and Lane Lines
    const camAngle = config.angle;
    const fovScale = 60 / config.fov;
    const centerX = width / 2;

    ctx.strokeStyle = currentLayer === SIM_CONFIG.LAYERS.SEMANTIC ? '#00FF88' : '#00F0FF';
    ctx.lineWidth = 1.5;

    // Projected Lane Lines
    const isLookingBack = Math.abs(camAngle) > Math.PI / 2;
    const zDir = isLookingBack ? 1 : -1;
    const dashOffset = ((carGroup.position.z * 1.5) % 30);

    for (let step = 10; step < config.range; step += 15) {
        const visualZ = step + (dashOffset * (isLookingBack ? -1 : 1));
        if (visualZ <= 2) continue;

        const projY = horizonY + (1.2 / visualZ) * (height * 0.8) * fovScale;
        if (projY > height || projY < horizonY) continue;

        const spread = (ROAD_WIDTH / visualZ) * (width * 0.45) * fovScale;
        const xOffset = -Math.sin(camAngle) * spread * 2.2;

        // Left & Right Road Edges
        ctx.fillStyle = 'rgba(0, 240, 255, 0.4)';
        ctx.fillRect(centerX + xOffset - spread, projY, 3, 2);
        ctx.fillRect(centerX + xOffset + spread, projY, 3, 2);

        // Dashed Center Lanes
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(centerX + xOffset - spread * 0.35, projY, 2, 4);
        ctx.fillRect(centerX + xOffset + spread * 0.35, projY, 2, 4);
    }

    // 4. Project 3D Obstacles & Dynamic Vehicles into this Camera's Frustum
    obstacleMeshes.forEach(obs => {
        const dx = obs.group.position.x - carGroup.position.x;
        const dz = obs.group.position.z - carGroup.position.z;

        // Transform into camera local coordinates (Rotate by camera yaw angle)
        const localX = dx * Math.cos(-camAngle) - dz * Math.sin(-camAngle);
        const localZ = dx * Math.sin(-camAngle) + dz * Math.cos(-camAngle);

        // Only render objects in front of this specific camera lens
        if (localZ < -1.5 && Math.abs(localZ) < config.range) {
            const depth = Math.abs(localZ);
            const screenX = centerX + (localX / depth) * (width * 0.8) * fovScale;
            const screenY = horizonY + (1.2 / depth) * (height * 0.8) * fovScale;

            const boxW = Math.max(8, (obs.data.size[0] / depth) * (width * 0.7) * fovScale);
            const boxH = Math.max(6, (obs.data.size[1] / depth) * (height * 0.7) * fovScale);

            if (screenX > -boxW && screenX < width + boxW && screenY > horizonY - boxH && screenY < height + boxH) {
                // Draw 3D Bounding Box
                let boxColor = '#00F0FF';
                if (obs.data.semantic === 'HAZARD') boxColor = '#FF003C';
                if (obs.data.semantic === 'PEDESTRIAN') boxColor = '#FFE600';

                ctx.strokeStyle = boxColor;
                ctx.lineWidth = 1.8;
                ctx.strokeRect(screenX - boxW/2, screenY - boxH, boxW, boxH);

                // AI Tag & Range Overlay
                ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
                ctx.fillRect(screenX - boxW/2, screenY - boxH - 12, boxW + 16, 11);

                ctx.fillStyle = boxColor;
                ctx.font = 'bold 8px JetBrains Mono';
                ctx.fillText(`${obs.data.label || 'OBJ'} ${depth.toFixed(0)}m`, screenX - boxW/2 + 2, screenY - boxH - 3);

                // ASTES Neuromorphic Micro-Polarity Events (Blue/Red sparks in CV-NF mode)
                if (currentMode === SIM_CONFIG.MODES.CVNF) {
                    for (let p = 0; p < 6; p++) {
                        const px = screenX - boxW/2 + Math.random() * boxW;
                        const py = screenY - boxH + Math.random() * boxH;
                        ctx.fillStyle = Math.random() > 0.5 ? '#00F0FF' : '#FF0055';
                        ctx.fillRect(px, py, 1.5, 1.5);
                    }
                }
            }
        }
    });

    // 5. Camera Info Overlays & Crosshairs
    ctx.fillStyle = 'rgba(0, 240, 255, 0.85)';
    ctx.font = '7.5px JetBrains Mono';
    ctx.fillText(config.label, 4, 11);

    ctx.fillStyle = '#00FF88';
    ctx.fillText(`${Math.round(speedKmh)} km/h | 1000Hz`, 4, height - 4);

    // Crosshair Center Target
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX - 8, horizonY); ctx.lineTo(centerX + 8, horizonY);
    ctx.moveTo(centerX, horizonY - 6); ctx.lineTo(centerX, horizonY + 6);
    ctx.stroke();
}

function updateSurroundCameras(speedKmh, now) {
    // 1. Render all 8 PiP sub-cameras
    Object.entries(cameraCanvases).forEach(([key, canvas]) => {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        renderSingleCameraView(ctx, canvas.width, canvas.height, key, speedKmh, now);
    });

    // 2. Render Modal Inspector if active
    if (isCameraModalOpen && modalCanvas) {
        const ctx = modalCanvas.getContext('2d');
        renderSingleCameraView(ctx, modalCanvas.width, modalCanvas.height, inspectedCamKey, speedKmh, now);
    }
}

// 10. Visual Layer Transformation Engine
function applyLayerVisuals(now) {
    if (currentLayer === SIM_CONFIG.LAYERS.NORMAL) {
        road.material.color.setHex(0x14171d);
        gridHelper.material.color.setHex(0x00f0ff);
        barrierMat.color.setHex(0x30363d);
        bodyMesh.material.color.setHex(0xd0d5dd);
        bodyMesh.material.wireframe = false;
        saliencyConeMat.opacity = 0.0;

        obstacleMeshes.forEach(obs => {
            obs.mesh.material.color.setHex(obs.data.color);
            obs.mesh.material.wireframe = false;
            obs.saliencyMesh.material.opacity = 0.0;
            obs.dangerDome.material.opacity = 0.0;
            obs.semanticShell.material.opacity = 0.0;
        });

    } else if (currentLayer === SIM_CONFIG.LAYERS.SALIENCY) {
        road.material.color.setHex(0x0b1120);
        gridHelper.material.color.setHex(0xff3b30);
        barrierMat.color.setHex(0xff0055);
        bodyMesh.material.color.setHex(0xff9500);
        bodyMesh.material.wireframe = false;
        
        saliencyCone.position.copy(carGroup.position);
        saliencyCone.position.z -= 25;
        saliencyConeMat.opacity = 0.35 * saliencyAlpha;

        obstacleMeshes.forEach(obs => {
            const dz = obs.group.position.z - carGroup.position.z;
            const dx = obs.group.position.x - carGroup.position.x;
            const dist = Math.sqrt(dx*dx + dz*dz);
            const attention = Math.min(1.0, (160.0 / Math.max(dist, 1.0))) * saliencyAlpha * obs.data.saliency;

            obs.mesh.material.color.setHex(0xff003c);
            obs.saliencyMesh.material.opacity = Math.max(0.2, attention * 0.9);
            obs.saliencyMesh.material.color.setHex(0xff3b30);
            obs.dangerDome.material.opacity = 0.0;
            obs.semanticShell.material.opacity = 0.0;
        });

    } else if (currentLayer === SIM_CONFIG.LAYERS.SEMANTIC) {
        road.material.color.setHex(0x063319);
        gridHelper.material.color.setHex(0x00ff88);
        barrierMat.color.setHex(0x990022);
        bodyMesh.material.color.setHex(0x00f0ff);
        bodyMesh.material.wireframe = true;
        saliencyConeMat.opacity = 0.0;

        obstacleMeshes.forEach(obs => {
            obs.saliencyMesh.material.opacity = 0.0;
            obs.dangerDome.material.opacity = 0.0;
            
            let semColor = 0x00f0ff;
            if (obs.data.semantic === 'HAZARD') semColor = 0xff003c;
            if (obs.data.semantic === 'PEDESTRIAN') semColor = 0xffe600;

            obs.mesh.material.color.setHex(semColor);
            obs.semanticShell.material.color.setHex(semColor);
            obs.semanticShell.material.opacity = 0.35;
        });

    } else if (currentLayer === SIM_CONFIG.LAYERS.UNCERTAINTY) {
        road.material.color.setHex(0x120826);
        gridHelper.material.color.setHex(0xbd00ff);
        barrierMat.color.setHex(0x7928ca);
        bodyMesh.material.color.setHex(0x00f0ff);
        bodyMesh.material.wireframe = false;
        saliencyConeMat.opacity = 0.0;

        const pulse = (Math.sin(now * 0.008) + 1.0) * 0.5;

        obstacleMeshes.forEach(obs => {
            obs.saliencyMesh.material.opacity = 0.0;
            obs.semanticShell.material.opacity = 0.0;

            if (obs.data.uncertainty >= uncertaintyThreshold) {
                obs.mesh.material.color.setHex(0xbd00ff);
                obs.mesh.material.wireframe = true;
                obs.dangerDome.material.opacity = 0.3 + pulse * 0.5;
            } else {
                obs.mesh.material.color.setHex(0x1a365d);
                obs.mesh.material.wireframe = false;
                obs.dangerDome.material.opacity = 0.0;
            }
        });
    }
}

// 11. Driving Physics & Keyboard State
const keys = { w: false, a: false, s: false, d: false, space: false };
let carSpeedKmh = 0;
let steerInput = 0;

window.addEventListener('keydown', (e) => {
    initAudio();
    const k = e.key.toLowerCase();
    if (k === 'w' || k === 'arrowup') keys.w = true;
    if (k === 's' || k === 'arrowdown') keys.s = true;
    if (k === 'a' || k === 'arrowleft') keys.a = true;
    if (k === 'd' || k === 'arrowright') keys.d = true;
    if (k === ' ') keys.space = true;
    if (k === 'e') {
        autoEvasionActive = !autoEvasionActive;
        const elAeb = document.getElementById('val-aeb');
        if (elAeb) {
            elAeb.innerText = autoEvasionActive ? "🟢 OTONOM KAÇIŞ DEVREDE" : "HAZIRDA (E Tuşu)";
            elAeb.style.color = autoEvasionActive ? "var(--tesla-green)" : "var(--tesla-cyan)";
        }
    }
    if (k === 'g') {
        targetLockActive = !targetLockActive;
        if (targetLockActive) {
            let bestTarget = null;
            let closestZ = -9999;
            obstacleMeshes.forEach(obs => {
                const dz = obs.group.position.z - carGroup.position.z;
                if (obs.data.isVehicle && dz < 0 && dz > -180 && dz > closestZ) {
                    closestZ = dz;
                    bestTarget = obs;
                }
            });

            if (bestTarget) {
                lockedTarget = bestTarget;
                playLockSound(true);
                targetLockGroup.visible = true;
                const elLock = document.getElementById('val-target-lock');
                if (elLock) {
                    elLock.innerText = `🎯 KİLİTLENDİ [HEDEF: ${bestTarget.data.type}]`;
                    elLock.style.color = "var(--tesla-green)";
                }
            } else {
                targetLockActive = false;
                targetLockGroup.visible = false;
            }
        } else {
            lockedTarget = null;
            targetLockGroup.visible = false;
            playLockSound(false);
            const elLock = document.getElementById('val-target-lock');
            if (elLock) {
                elLock.innerText = "SERBEST (Kilit Yok)";
                elLock.style.color = "var(--text-secondary)";
            }
        }
    }
    if (k === 'c') {
        cameraView = cameraView === 'CHASE' ? 'COCKPIT' : 'CHASE';
    }
});

window.addEventListener('keyup', (e) => {
    const k = e.key.toLowerCase();
    if (k === 'w' || k === 'arrowup') keys.w = false;
    if (k === 's' || k === 'arrowdown') keys.s = false;
    if (k === 'a' || k === 'arrowleft') keys.a = false;
    if (k === 'd' || k === 'arrowright') keys.d = false;
    if (k === ' ') keys.space = false;
});

// Mode, Weather, Layer & Scoreboard
const btnStd = document.getElementById('btn-mode-std');
const btnCvnf = document.getElementById('btn-mode-cvnf');

function setMode(mode) {
    currentMode = mode;
    if (mode === SIM_CONFIG.MODES.STANDARD) {
        btnStd.classList.add('active');
        btnCvnf.classList.remove('active');
    } else {
        btnCvnf.classList.add('active');
        btnStd.classList.remove('active');
    }
}
btnStd.addEventListener('click', () => setMode(SIM_CONFIG.MODES.STANDARD));
btnCvnf.addEventListener('click', () => setMode(SIM_CONFIG.MODES.CVNF));

window.setWeather = function(weatherType) {
    currentWeather = weatherType;
    ['clear', 'fog', 'rain'].forEach(w => {
        const btn = document.getElementById(`btn-weather-${w}`);
        if (btn) {
            btn.style.border = (w.toUpperCase() === weatherType) ? "1px solid var(--tesla-cyan)" : "1px solid var(--panel-border)";
            btn.style.background = (w.toUpperCase() === weatherType) ? "rgba(0, 240, 255, 0.25)" : "rgba(0,0,0,0.5)";
            btn.style.color = (w.toUpperCase() === weatherType) ? "#FFF" : "var(--text-secondary)";
        }
    });

    if (weatherType === SIM_CONFIG.WEATHER.FOG) {
        scene.fog.density = 0.045;
        rainMat.opacity = 0.0;
    } else if (weatherType === SIM_CONFIG.WEATHER.RAIN) {
        scene.fog.density = 0.025;
        rainMat.opacity = 0.7;
    } else {
        scene.fog.density = 0.010;
        rainMat.opacity = 0.0;
    }
};

window.setVisualLayer = function(layerName) {
    currentLayer = layerName;
    ['normal', 'saliency', 'semantic', 'uncertainty'].forEach(name => {
        const btn = document.getElementById(`btn-layer-${name}`);
        if (btn) {
            const isActive = (name.toUpperCase() === layerName);
            btn.style.border = isActive ? "1px solid var(--tesla-cyan)" : "1px solid var(--panel-border)";
            btn.style.background = isActive ? "rgba(0, 240, 255, 0.35)" : "rgba(0,0,0,0.5)";
            btn.style.color = isActive ? "#FFFFFF" : "var(--text-secondary)";
            btn.style.fontWeight = isActive ? "bold" : "normal";
        }
    });
    applyLayerVisuals(performance.now());
};

window.openScoreboard = function() {
    document.getElementById('safety-modal').style.display = 'flex';
};

window.closeScoreboard = function() {
    document.getElementById('safety-modal').style.display = 'none';
};

// Sliders
const sliderSaliency = document.getElementById('slider-saliency');
if (sliderSaliency) {
    sliderSaliency.addEventListener('input', (e) => {
        saliencyAlpha = parseFloat(e.target.value);
        const lbl = document.getElementById('lbl-saliency-val');
        if (lbl) lbl.innerText = `${Math.round(saliencyAlpha * 100)}%`;
        applyLayerVisuals(performance.now());
    });
}

const sliderUncertainty = document.getElementById('slider-uncertainty');
if (sliderUncertainty) {
    sliderUncertainty.addEventListener('input', (e) => {
        uncertaintyThreshold = parseFloat(e.target.value);
        const lbl = document.getElementById('lbl-uncertainty-val');
        if (lbl) lbl.innerText = uncertaintyThreshold.toFixed(2);
        applyLayerVisuals(performance.now());
    });
}

window.setMission = function(missionId) {
    currentMission = missionId;
    document.querySelectorAll('.mission-card').forEach((card, idx) => {
        card.classList.toggle('active', (idx + 1) === missionId);
    });

    if (missionId === 1) {
        carGroup.position.set(0, 0, 30);
        carSpeedKmh = 70;
    } else if (missionId === 2) {
        carGroup.position.set(0, 0, -350);
        carSpeedKmh = 100;
    } else if (missionId === 3) {
        carGroup.position.set(-3.5, 0, -800);
        carSpeedKmh = 90;
    }

    carGroup.rotation.y = 0;
    steerInput = 0;
    camera.position.set(carGroup.position.x, 4.8, carGroup.position.z + 11.0);
    camera.lookAt(carGroup.position.x, 1.3, carGroup.position.z - 30);
};

// 12. Main Physics, Target Lock Tracking & Animation Loop
let lastFrameTime = performance.now();
let v2xAlertPlayed = false;

function animate(now) {
    requestAnimationFrame(animate);
    const dt = Math.min((now - lastFrameTime) / 1000, 0.1);
    lastFrameTime = now;

    // A. LiDAR Spinner & Reticle Spin
    lidarSpinner.rotation.y += 25.0 * dt;
    reticleRing.rotation.z += 3.0 * dt;

    // B. Rain Particles
    if (currentWeather === SIM_CONFIG.WEATHER.RAIN) {
        const pos = rainGeo.attributes.position.array;
        for (let i = 1; i < rainCount * 3; i += 3) {
            pos[i] -= 45.0 * dt;
            if (pos[i] < 0) pos[i] = 30.0;
        }
        rainGeo.attributes.position.needsUpdate = true;
        rainParticles.position.z = carGroup.position.z;
    }

    // C. Direksiyon & Otonom Hedef Takibi (G Tuşu)
    let targetSteer = 0;
    if (keys.a) targetSteer -= 1.0; // A: SOLA DÖN (-X)
    if (keys.d) targetSteer += 1.0; // D: SAĞA DÖN (+X)

    if (targetLockActive && lockedTarget) {
        const targetX = lockedTarget.group.position.x;
        const deltaX = targetX - carGroup.position.x;
        targetSteer = THREE.MathUtils.clamp(deltaX * 0.8, -1.0, 1.0);

        const targetZ = lockedTarget.group.position.z;
        const currentDist = Math.abs(targetZ - carGroup.position.z);
        const distError = currentDist - SIM_CONFIG.TARGET_FOLLOW_DIST;

        if (distError > 2.0) {
            carSpeedKmh = Math.min(carSpeedKmh + SIM_CONFIG.ACCEL * 45 * dt, 110);
        } else if (distError < -1.0) {
            carSpeedKmh = Math.max(carSpeedKmh - SIM_CONFIG.BRAKE * 45 * dt, 10);
        }

        targetLockGroup.position.copy(lockedTarget.group.position);
        targetLockGroup.position.y += 1.5;
    } else {
        targetLockGroup.visible = false;
    }

    // Otonom Kaçış Mantığı (E Tuşu)
    if (autoEvasionActive) {
        obstacleMeshes.forEach(obs => {
            const dz = obs.group.position.z - carGroup.position.z;
            const dx = obs.group.position.x - carGroup.position.x;
            if (dz < 0 && Math.abs(dz) < 45.0 && Math.abs(dx) < 2.5) {
                // Sağdaysak sola kaç (-1.0), soldaysak sağa kaç (+1.0)
                targetSteer = carGroup.position.x > 0 ? -1.0 : 1.0;
                playRadarPing(0, 1100, 0.08);
            }
        });
    }

    steerInput = THREE.MathUtils.lerp(steerInput, targetSteer, dt * 8.0);

    const wheelSteerRad = -steerInput * 0.25;
    frontLeftPivot.rotation.y = wheelSteerRad;
    frontRightPivot.rotation.y = wheelSteerRad;

    const bodyRollYaw = -steerInput * SIM_CONFIG.MAX_STEER_ANGLE;
    carGroup.rotation.y = bodyRollYaw;

    // D. Tesla EV Doğrudan Tahrik (İleri [D] & Geri [R] Sürüş)
    if (!targetLockActive) {
        if (keys.w) {
            if (carSpeedKmh < -0.5) {
                // Geri vitesteyken W basılırsa önce fren yapıp dur
                carSpeedKmh = Math.min(carSpeedKmh + SIM_CONFIG.BRAKE * 60 * dt, 0);
            } else {
                // İleri hızlan
                carSpeedKmh = Math.min(carSpeedKmh + SIM_CONFIG.ACCEL * 60 * dt, SIM_CONFIG.MAX_SPEED_KMH);
            }
        }
        if (keys.s) {
            if (carSpeedKmh > 0.5) {
                // İleri giderken S basılırsa fren yap
                carSpeedKmh = Math.max(carSpeedKmh - SIM_CONFIG.BRAKE * 60 * dt, 0);
            } else {
                // 0 veya gerideyken S basılırsa GERİ GERİ HIZLAN (Max -45 km/s)
                carSpeedKmh = Math.max(carSpeedKmh - SIM_CONFIG.ACCEL * 45 * dt, -45);
            }
        }
        if (keys.space) carSpeedKmh *= Math.pow(0.70, dt * 60);
        if (!keys.w && !keys.s) carSpeedKmh *= Math.pow(SIM_CONFIG.FRICTION, dt * 60);
    }

    const speedMps = (carSpeedKmh * 1000) / 3600;

    // Geri Vites Beyaz Işığı (Reverse Lights)
    if (carSpeedKmh < -0.5) {
        rearLight.material.color.setHex(0xffffff); // Beyaz geri lambası
    } else {
        rearLight.material.color.setHex(0xe82127); // Kırmızı fren/stop lambası
    }

    // E. Tekerlek Dönüşü (İleri & Geri Senkron)
    const wheelRollDelta = (speedMps / 0.45) * dt;
    frontLeftWheel.rotation.x -= wheelRollDelta;
    frontRightWheel.rotation.x -= wheelRollDelta;
    rearLeftWheel.rotation.x -= wheelRollDelta;
    rearRightWheel.rotation.x -= wheelRollDelta;

    // F. Boylamsal & Yanal İlerleme
    carGroup.position.z -= speedMps * dt;
    if (Math.abs(speedMps) > 0.1) {
        const steerDir = speedMps > 0 ? 1 : -1;
        carGroup.position.x += steerInput * steerDir * SIM_CONFIG.LANE_SLIDE_SPEED * dt;
    }

    const maxRoadX = ROAD_WIDTH / 2 - 2.2;
    carGroup.position.x = Math.max(-maxRoadX, Math.min(maxRoadX, carGroup.position.x));

    // Çift Yönlü Yol Sonsuz Döngüsü (İleri & Geri)
    if (carGroup.position.z < -ROAD_LENGTH + 100) {
        carGroup.position.z = 40;
    } else if (carGroup.position.z > 60) {
        carGroup.position.z = -ROAD_LENGTH + 120;
    }

    // G. 360° LiDAR, Katman Dönüşümü ve Kör Nokta Tehdit Analizi
    let minObstacleDist = 999.0;
    let nearestObs = null;
    let leftHazard = false;
    let rightHazard = false;

    obstacleMeshes.forEach(obs => {
        if (obs.data.speed) {
            obs.group.position.z += obs.data.speed;
            if (obs.group.position.z < -ROAD_LENGTH + 100) obs.group.position.z = -50;
        }

        const dz = obs.group.position.z - carGroup.position.z;
        const dx = obs.group.position.x - carGroup.position.x;
        const dist = Math.sqrt(dx*dx + dz*dz);

        // Önümüzdeki engel
        if (dz < 0 && Math.abs(dz) < minObstacleDist) {
            minObstacleDist = Math.abs(dz);
            nearestObs = obs;
        }

        // Kör Nokta Tespiti
        if (Math.abs(dz) < 12.0 && dist < 14.0) {
            if (dx < -1.5) {
                leftHazard = true;
                playRadarPing(-0.8, 650, 0.05);
            }
            if (dx > 1.5) {
                rightHazard = true;
                playRadarPing(0.8, 650, 0.05);
            }
        }
    });

    // 🛡️ HJI Reachability Barrier h(x) ve Acil Tünel Tehlike Alarmı
    // h(x) = (Mesafe - Güvenlik Yarıçapı)
    const hjiBarrierVal = (minObstacleDist - 12.0) / 4.0;
    const elHjiHud = document.getElementById('val-hji-barrier');
    const collisionBanner = document.getElementById('collision-alert-banner');
    const collisionText = document.getElementById('collision-alert-text');

    const inTunnelZone = (carGroup.position.z < -450 && carGroup.position.z > -750);
    const isCriticalNear = minObstacleDist < 35.0;

    // Tünel Kırmızı Alarm Aydınlatması
    if (inTunnelZone && isCriticalNear) {
        tunnelMesh.material.color.setHex(0x33050a); // Kırmızı alarm tünel duvarı
        tunnelLightSources.forEach(l => {
            l.color.setHex(0xff003c); // Kırmızı flaşör tavan ışıkları
            l.intensity = 3.5;
        });
        if (nearestObs) {
            nearestObs.mesh.material.color.setHex(0xff003c); // Engeli parlayan kırmızı yap
        }
    } else {
        tunnelMesh.material.color.setHex(0x090b0e);
        tunnelLightSources.forEach(l => {
            l.color.setHex(0xffb800); // Normal amber tünel ışığı
            l.intensity = 2.5;
        });
    }

    // HJI Güvenlik Tüpü Renkleri & Acil Alarm Bannerı
    if (minObstacleDist < 30.0) {
        // 🔴 KRİTİK ÇARPIŞMA RİSKİ / HJI MÜDAHALE
        const pulse = (Math.sin(now * 0.02) + 1.0) * 0.5;
        hjiTubeMat.color.setHex(0xff003c); // Kırmızı tehlike tüpü
        hjiTubeMat.opacity = 0.45 + pulse * 0.45;
        
        if (elHjiHud) {
            elHjiHud.innerText = `${hjiBarrierVal.toFixed(2)} m (MÜDAHALE!)`;
            elHjiHud.style.color = "#FF7B72";
        }

        if (collisionBanner) {
            collisionBanner.style.display = 'flex';
            if (collisionText) {
                collisionText.innerText = inTunnelZone 
                    ? `🚨 TÜNELDE ACİL ENGEL: ${minObstacleDist.toFixed(1)}m İLERİDE! FREN YAPIN VEYA 'E' İLE KAÇIN!`
                    : `🚨 ACİL ÇARPIŞMA UYARISI: ${minObstacleDist.toFixed(1)}m İLERİDE ENGEL!`;
            }
        }

        // Sesli Uyarı
        if (Math.random() < 0.25) {
            playRadarPing(0, 1500, 0.1);
        }

    } else if (minObstacleDist < 60.0) {
        // 🟡 YAKLAŞMA UYARISI
        hjiTubeMat.color.setHex(0xffb800); // Sarı uyarı tüpü
        hjiTubeMat.opacity = 0.38;

        if (elHjiHud) {
            elHjiHud.innerText = `+${hjiBarrierVal.toFixed(2)} m (SINIRDA)`;
            elHjiHud.style.color = "#FFB800";
        }
        if (collisionBanner) collisionBanner.style.display = 'none';

    } else {
        // 🟢 GÜVENLİ İNVARYANT KORİDOR
        hjiTubeMat.color.setHex(0x00ff88);
        hjiTubeMat.opacity = 0.22;

        if (elHjiHud) {
            elHjiHud.innerText = `+${hjiBarrierVal.toFixed(2)} m (KORUMALI)`;
            elHjiHud.style.color = "var(--tesla-green)";
        }
        if (collisionBanner) collisionBanner.style.display = 'none';
    }

    applyLayerVisuals(now);

    leftBlindMat.opacity = leftHazard ? 0.7 : 0.0;
    rightBlindMat.opacity = rightHazard ? 0.7 : 0.0;

    // H. V2X Filo Ağı
    const distToTunnel = -450 - carGroup.position.z;
    const v2xBanner = document.getElementById('v2x-alert-banner');
    if (distToTunnel > 0 && distToTunnel < 380) {
        if (v2xBanner) v2xBanner.style.display = 'flex';
        if (!v2xAlertPlayed) {
            playRadarPing(0, 1300, 0.25);
            v2xAlertPlayed = true;
        }
    } else {
        if (v2xBanner) v2xBanner.style.display = 'none';
        v2xAlertPlayed = false;
    }

    // I. Telemetri HUD
    const elSpeed = document.getElementById('val-speed');
    const elTtc = document.getElementById('val-ttc');
    const elBlindSpot = document.getElementById('val-blindspot');

    if (elSpeed) {
        if (carSpeedKmh < -0.5) {
            elSpeed.innerText = `R ${Math.abs(Math.round(carSpeedKmh))}`;
            elSpeed.style.color = "#FFB800"; // Geri viteste amber renk
        } else {
            elSpeed.innerText = Math.round(carSpeedKmh);
            elSpeed.style.color = "var(--tesla-cyan)";
        }
    }

    if (elTtc) {
        const absSpeedMps = Math.abs(speedMps);
        const ttcSeconds = absSpeedMps > 1.0 ? (minObstacleDist / absSpeedMps).toFixed(1) : "∞";
        elTtc.innerText = `${minObstacleDist.toFixed(1)} m (${ttcSeconds}s)`;
        elTtc.style.color = minObstacleDist < 30 ? "#FF7B72" : "var(--tesla-cyan)";
    }

    if (elBlindSpot) {
        if (leftHazard) {
            elBlindSpot.innerText = "⚠️ SOL KÖR NOKTA (Araç Yaklaşıyor!)";
            elBlindSpot.style.color = "#FF7B72";
        } else if (rightHazard) {
            elBlindSpot.innerText = "⚠️ SAĞ KÖR NOKTA (Araç Yaklaşıyor!)";
            elBlindSpot.style.color = "#FF7B72";
        } else {
            elBlindSpot.innerText = "TEMİZ (Tehlike Yok)";
            elBlindSpot.style.color = "var(--tesla-green)";
        }
    }

    // Kamera Takibi
    if (cameraView === 'CHASE') {
        const targetCamX = THREE.MathUtils.lerp(camera.position.x, carGroup.position.x, dt * 7.0);
        const targetCamZ = carGroup.position.z + 10.5;
        camera.position.set(targetCamX, 4.6, targetCamZ);
        camera.lookAt(carGroup.position.x, 1.4, carGroup.position.z - 25);
    } else {
        camera.position.set(carGroup.position.x, 1.45, carGroup.position.z - 0.4);
        camera.lookAt(carGroup.position.x, 1.45, carGroup.position.z - 40);
    }

    // 📹 8-Kamera Surround Projeksiyon Motorunu Çalıştır
    updateSurroundCameras(carSpeedKmh, now);

    renderer.render(scene, camera);
}

// Window Resize Handling
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Start Animation Loop
animate(performance.now());
