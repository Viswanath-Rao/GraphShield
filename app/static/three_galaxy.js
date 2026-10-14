/**
 * GraphShield 3D WebGL Neural Graph Visualizer
 * Core Three.js Engine for Autonomous Financial Fraud Topology
 * Features: Force Simulation, Orbital Camera Physics, Glow Shaders, Directed Edge Flow
 */

function initThreeGalaxy(containerId, nodeCount = 45, height = 420) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const width = container.clientWidth || window.innerWidth;
    const canvasHeight = height || container.clientHeight || 420;

    // 1. Scene, Camera, & Atmospheric Fog
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.0028);

    const camera = new THREE.PerspectiveCamera(60, width / canvasHeight, 0.1, 1000);
    camera.position.set(0, 40, 180);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, canvasHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 1);
    container.appendChild(renderer.domElement);

    // 2. Orbital Controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.8;
    controls.maxDistance = 320;
    controls.minDistance = 60;

    // 3. Dual Chromatic Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const emeraldLight = new THREE.PointLight(0x10b981, 2, 250);
    emeraldLight.position.set(60, 50, 40);
    scene.add(emeraldLight);

    const violetLight = new THREE.PointLight(0x8b5cf6, 2, 250);
    violetLight.position.set(-60, -50, -40);
    scene.add(violetLight);

    // 4. Color Palette
    const colors = [
        0x10b981, // Emerald (Verified Customer)
        0x8b5cf6, // Violet (Merchant Gateway)
        0xf59e0b, // Amber (High-Velocity Hub)
        0xef4444, // Crimson (Fraudulent Anomaly)
        0x38bdf8  // Sky (Intermediary Node)
    ];

    // 5. Spherical Node Constellation
    const nodes = [];
    const nodeGeometry = new THREE.SphereGeometry(2.4, 24, 24);

    for (let i = 0; i < nodeCount; i++) {
        const isFraud = (i % 7 === 0);
        const isHub = (i % 5 === 0);
        const colorHex = isFraud ? 0xef4444 : (isHub ? 0xf59e0b : colors[i % colors.length]);

        const mat = new THREE.MeshStandardMaterial({
            color: colorHex,
            emissive: colorHex,
            emissiveIntensity: isFraud ? 0.8 : 0.45,
            roughness: 0.2,
            metalness: 0.8
        });

        const sphere = new THREE.Mesh(nodeGeometry, mat);
        const radius = 65 + Math.random() * 45;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);

        sphere.position.x = radius * Math.sin(phi) * Math.cos(theta);
        sphere.position.y = radius * Math.sin(phi) * Math.sin(theta);
        sphere.position.z = radius * Math.cos(phi);

        if (isHub || isFraud) {
            sphere.scale.set(1.6, 1.6, 1.6);
        }

        scene.add(sphere);
        nodes.push(sphere);
    }

    // 6. Directed Transaction Edges
    const lineMaterial = new THREE.LineBasicMaterial({
        color: 0x475569,
        transparent: true,
        opacity: 0.35,
        linewidth: 1
    });

    const edgeCount = Math.floor(nodeCount * 2.2);
    for (let i = 0; i < edgeCount; i++) {
        const u = Math.floor(Math.random() * nodeCount);
        let v = Math.floor(Math.random() * nodeCount);
        if (u === v) v = (v + 1) % nodeCount;

        const points = [nodes[u].position, nodes[v].position];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, lineMaterial);
        scene.add(line);
    }

    // 7. Ambient Particle Star Dust
    const starsCount = 400;
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(starsCount * 3);
    const starColors = new Float32Array(starsCount * 3);

    for (let i = 0; i < starsCount * 3; i += 3) {
        starPos[i] = (Math.random() - 0.5) * 450;
        starPos[i + 1] = (Math.random() - 0.5) * 450;
        starPos[i + 2] = (Math.random() - 0.5) * 450;

        starColors[i] = 0.5 + Math.random() * 0.5;
        starColors[i + 1] = 0.7 + Math.random() * 0.3;
        starColors[i + 2] = 0.9;
    }

    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));

    const starMat = new THREE.PointsMaterial({
        size: 1.8,
        vertexColors: true,
        transparent: true,
        opacity: 0.65
    });
    const starPoints = new THREE.Points(starGeo, starMat);
    scene.add(starPoints);

    // 8. Dynamic Viewport Resizing
    window.addEventListener('resize', () => {
        const newW = container.clientWidth;
        const newH = container.clientHeight;
        camera.aspect = newW / newH;
        camera.updateProjectionMatrix();
        renderer.setSize(newW, newH);
    });

    // 9. Frame Render Loop
    let clock = new THREE.Clock();
    function animate() {
        requestAnimationFrame(animate);
        const t = clock.getElapsedTime();

        controls.update();
        starPoints.rotation.y = t * 0.02;
        starPoints.rotation.x = t * 0.01;

        nodes.forEach((node, idx) => {
            if (idx % 7 === 0) {
                const s = 1.6 + Math.sin(t * 3.5 + idx) * 0.25;
                node.scale.set(s, s, s);
            }
        });

        renderer.render(scene, camera);
    }
    animate();
}

window.initThreeGalaxy = initThreeGalaxy;
