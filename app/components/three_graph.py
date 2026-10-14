"""
GraphShield 3D WebGL Neural Graph Visualizer (Three.js)
Renders a live 3D force-directed financial transaction cluster with orbital controls,
glowing vertices, transaction particle streams, and cybernetic HUD overlays.
"""

def get_threejs_graph_html(num_nodes: int = 40, height: int = 420) -> str:
    """
    Returns self-contained HTML/JS using Three.js to render an interactive 3D WebGL graph.
    Uses clean string substitution to avoid Python f-string bracket escaping conflicts.
    """
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background-color: #000000;
                overflow: hidden;
                width: 100%;
                height: __HEIGHT__px;
                font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
            }
            #webgl-container {
                width: 100%;
                height: 100%;
                position: relative;
            }
            .hud-overlay {
                position: absolute;
                top: 14px;
                left: 18px;
                pointer-events: none;
                z-index: 10;
            }
            .hud-title {
                font-size: 0.85rem;
                font-weight: 800;
                color: #ffffff;
                letter-spacing: 1.5px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .hud-pill {
                font-family: monospace;
                font-size: 0.65rem;
                color: #10b981;
                background: rgba(16, 185, 129, 0.15);
                border: 1px solid rgba(16, 185, 129, 0.4);
                padding: 2px 8px;
                border-radius: 4px;
                margin-top: 4px;
                display: inline-block;
            }
            .hud-controls {
                position: absolute;
                bottom: 12px;
                right: 16px;
                font-size: 0.7rem;
                color: #94a3b8;
                font-family: monospace;
                pointer-events: none;
                background: rgba(8, 8, 14, 0.75);
                padding: 4px 10px;
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="webgl-container">
            <div class="hud-overlay">
                <div class="hud-title">🌐 3D WEBGL GRAPH TOPOLOGY</div>
                <div class="hud-pill">GNN SPECTRAL EMBEDDING • LIVE FORCE SIMULATION</div>
            </div>
            <div class="hud-controls">🖱️ DRAG TO ROTATE • SCROLL TO ZOOM</div>
        </div>

        <script>
            const container = document.getElementById('webgl-container');
            const width = container.clientWidth;
            const height = container.clientHeight;

            // Scene, Camera, Renderer
            const scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x000000, 0.0028);

            const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
            camera.position.set(0, 40, 180);

            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(width, height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.setClearColor(0x000000, 1);
            container.appendChild(renderer.domElement);

            // Controls
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.06;
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.8;
            controls.maxDistance = 320;
            controls.minDistance = 60;

            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);

            const pointLight1 = new THREE.PointLight(0x10b981, 2, 250);
            pointLight1.position.set(60, 50, 40);
            scene.add(pointLight1);

            const pointLight2 = new THREE.PointLight(0x8b5cf6, 2, 250);
            pointLight2.position.set(-60, -50, -40);
            scene.add(pointLight2);

            // Palette
            const colors = [
                0x10b981, // Emerald (Verified Customer)
                0x8b5cf6, // Violet (Merchant Gateway)
                0xf59e0b, // Amber (High-Velocity Hub)
                0xef4444, // Crimson (Fraudulent Node)
                0x38bdf8  // Sky (Intermediary Account)
            ];

            // Generate Node Clusters
            const nodeCount = __NUM_NODES__;
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

                // Scale hubs
                if (isHub || isFraud) {
                    sphere.scale.set(1.6, 1.6, 1.6);
                }

                scene.add(sphere);
                nodes.push(sphere);
            }

            // Generate Directed Transaction Edges
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

            // Ambient Floating Star Dust / Background Particles
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

            // Responsive Resize
            window.addEventListener('resize', () => {
                const newW = container.clientWidth;
                const newH = container.clientHeight;
                camera.aspect = newW / newH;
                camera.updateProjectionMatrix();
                renderer.setSize(newW, newH);
            });

            // Animation Loop
            let clock = new THREE.Clock();
            function animate() {
                requestAnimationFrame(animate);
                const t = clock.getElapsedTime();

                controls.update();
                starPoints.rotation.y = t * 0.02;
                starPoints.rotation.x = t * 0.01;

                // Gentle pulsing for fraud nodes
                nodes.forEach((node, idx) => {
                    if (idx % 7 === 0) {
                        const s = 1.6 + Math.sin(t * 3.5 + idx) * 0.25;
                        node.scale.set(s, s, s);
                    }
                });

                renderer.render(scene, camera);
            }
            animate();
        </script>
    </body>
    </html>
    """
    return template.replace("__NUM_NODES__", str(num_nodes)).replace("__HEIGHT__", str(height))
