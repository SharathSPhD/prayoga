import { useEffect, useRef } from 'preact/hooks';
import * as THREE from 'three';

interface Props {
  height: number;
}

export default function ParticleBrain({ height }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0);
    containerRef.current.appendChild(renderer.domElement);

    camera.position.z = 30;

    // Create particles
    const particleCount = 1500;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const velocities = new Float32Array(particleCount * 3);

    // Refusal "direction" along z-axis
    const refusalAxis = new THREE.Vector3(0, 0, 1);

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      positions[i3] = (Math.random() - 0.5) * 60;
      positions[i3 + 1] = (Math.random() - 0.5) * 60;
      positions[i3 + 2] = (Math.random() - 0.5) * 30;

      velocities[i3] = (Math.random() - 0.5) * 0.1;
      velocities[i3 + 1] = (Math.random() - 0.5) * 0.1;
      velocities[i3 + 2] = (Math.random() - 0.5) * 0.05;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));

    const material = new THREE.PointsMaterial({
      color: 0x22d3ee,
      size: 0.5,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.8,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // Animate
    let injectionPhase = 0;
    const animate = () => {
      requestAnimationFrame(animate);

      injectionPhase += 0.002;
      const injectionStrength = Math.sin(injectionPhase) * 0.5 + 0.5;

      const positionAttribute = geometry.getAttribute('position') as THREE.BufferAttribute;
      const velocityAttribute = geometry.getAttribute('velocity') as THREE.BufferAttribute;
      const pos = positionAttribute.array as Float32Array;
      const vel = velocityAttribute.array as Float32Array;

      for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;

        // Apply symmetry-breaking "injection" perturbation
        const perturbMag = injectionStrength * 0.01;
        pos[i3] += vel[i3] + (Math.random() - 0.5) * perturbMag;
        pos[i3 + 1] += vel[i3 + 1] + (Math.random() - 0.5) * perturbMag;
        pos[i3 + 2] += vel[i3 + 2];

        // Damping
        vel[i3] *= 0.98;
        vel[i3 + 1] *= 0.98;
        vel[i3 + 2] *= 0.98;

        // Bounds check
        if (Math.abs(pos[i3]) > 30) vel[i3] *= -1;
        if (Math.abs(pos[i3 + 1]) > 30) vel[i3 + 1] *= -1;
      }

      positionAttribute.needsUpdate = true;

      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      const newWidth = containerRef.current?.clientWidth || width;
      camera.aspect = newWidth / height;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, height);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      containerRef.current?.removeChild(renderer.domElement);
    };

  }, [height]);

  return <div ref={containerRef} style={{ width: '100%', height }} />;
}
