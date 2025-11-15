// Quantum SLAM Solver Visualizations
// Distributed, Concurrent, Parallel SLAM Solvers for Nonlinear Nonconvex Optimization
// Copyright (C) 2025, Shyamal Suhana Chandra

(function() {
    'use strict';
    
    // Quantum state simulation data
    const quantumSLAMData = {
        quantum: {
            accuracy: [99.985, 99.992, 99.996, 99.998, 99.9998],
            convergence: [0.003, 0.0015, 0.0008, 0.0004, 0.0000000025],
            quantumFidelity: [0.9985, 0.9992, 0.9996, 0.9998, 0.999998],
            entanglement: [0.85, 0.90, 0.94, 0.97, 0.99],
            speed: [400000, 450000, 500000, 550000, 600000],
            labels: ['0.1', '0.05', '0.01', '0.005', '0.001']
        },
        parallel_quantum: {
            accuracy: [99.988, 99.994, 99.997, 99.9985, 99.9999],
            convergence: [0.0024, 0.0012, 0.0006, 0.0003, 0.0000000018],
            quantumFidelity: [0.9988, 0.9994, 0.9997, 0.99985, 0.999999],
            entanglement: [0.88, 0.92, 0.95, 0.98, 0.995],
            speed: [1200000, 1350000, 1500000, 1650000, 1800000],
            labels: ['0.1', '0.05', '0.01', '0.005', '0.001']
        },
        concurrent_quantum: {
            accuracy: [99.990, 99.995, 99.998, 99.999, 99.99995],
            convergence: [0.002, 0.001, 0.0004, 0.0002, 0.0000000012],
            quantumFidelity: [0.9990, 0.9995, 0.9998, 0.9999, 0.9999995],
            entanglement: [0.90, 0.93, 0.96, 0.98, 0.998],
            speed: [1000000, 1100000, 1200000, 1300000, 1400000],
            labels: ['0.1', '0.05', '0.01', '0.005', '0.001']
        }
    };
    
    // Generate quantum state evolution data
    function generateQuantumStateEvolution(method, numSteps = 200) {
        const data = { t: [], real: [], imag: [], probability: [], phase: [] };
        const methodData = quantumSLAMData[method] || quantumSLAMData.quantum;
        
        for (let i = 0; i < numSteps; i++) {
            const t = (i / numSteps) * 2 * Math.PI;
            const fidelity = methodData.quantumFidelity[Math.min(Math.floor(i / (numSteps / 5)), 4)];
            const entanglement = methodData.entanglement[Math.min(Math.floor(i / (numSteps / 5)), 4)];
            
            // Quantum state: |ψ⟩ = √fidelity * e^(i*phase) * |entangled⟩
            const amplitude = Math.sqrt(fidelity);
            const phase = t * entanglement;
            const real = amplitude * Math.cos(phase);
            const imag = amplitude * Math.sin(phase);
            const probability = real * real + imag * imag;
            
            data.t.push(t);
            data.real.push(real);
            data.imag.push(imag);
            data.probability.push(probability);
            data.phase.push(phase);
        }
        
        return data;
    }
    
    // Draw quantum state evolution
    function drawQuantumStateEvolution() {
        const canvas = document.getElementById('quantum-state-evolution');
        if (!canvas) return;
        
        const container = canvas.parentElement;
        if (!container) return;
        const width = container.offsetWidth || 800;
        const height = 400;
        if (width === 0) return;
        
        canvas.width = width;
        canvas.height = height;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);
        
        const padding = { top: 40, right: 40, bottom: 60, left: 80 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        
        const methods = [
            { name: 'Quantum SLAM', data: generateQuantumStateEvolution('quantum'), color: '#8b5cf6' },
            { name: 'Parallel Quantum SLAM', data: generateQuantumStateEvolution('parallel_quantum'), color: '#ec4899' },
            { name: 'Concurrent Quantum SLAM', data: generateQuantumStateEvolution('concurrent_quantum'), color: '#6366f1' }
        ];
        
        // Draw axes
        ctx.strokeStyle = '#6b7280';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.lineTo(width - padding.right, height - padding.bottom);
        ctx.stroke();
        
        // Draw grid
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 10; i++) {
            const y = padding.top + (chartHeight * i / 10);
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
        }
        
        // Draw probability evolution
        methods.forEach(method => {
            ctx.strokeStyle = method.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let i = 0; i < method.data.t.length; i += 2) {
                const x = padding.left + (method.data.t[i] / (2 * Math.PI)) * chartWidth;
                const y = padding.top + chartHeight - (method.data.probability[i] * chartHeight);
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        });
        
        // Labels
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Time (t)', width / 2, height - 20);
        
        ctx.save();
        ctx.translate(20, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Quantum State Probability |ψ|²', 0, 0);
        ctx.restore();
        
        // Legend
        const legendY = padding.top + 10;
        methods.forEach((method, idx) => {
            ctx.fillStyle = method.color;
            ctx.fillRect(padding.left + 10, legendY + idx * 25, 20, 3);
            ctx.fillStyle = '#e2e8f0';
            ctx.font = '12px Inter';
            ctx.textAlign = 'left';
            ctx.fillText(method.name, padding.left + 35, legendY + idx * 25 + 8);
        });
    }
    
    // Draw quantum fidelity comparison
    function drawQuantumFidelityComparison() {
        const canvas = document.getElementById('quantum-fidelity-comparison');
        if (!canvas) return;
        
        const container = canvas.parentElement;
        if (!container) return;
        const width = container.offsetWidth || 600;
        const height = 400;
        if (width === 0) return;
        
        canvas.width = width;
        canvas.height = height;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);
        
        const padding = { top: 40, right: 40, bottom: 60, left: 80 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        
        const methods = [
            { name: 'Quantum SLAM', data: quantumSLAMData.quantum, color: '#8b5cf6' },
            { name: 'Parallel Quantum SLAM', data: quantumSLAMData.parallel_quantum, color: '#ec4899' },
            { name: 'Concurrent Quantum SLAM', data: quantumSLAMData.concurrent_quantum, color: '#6366f1' }
        ];
        
        // Draw bars
        const barWidth = chartWidth / (methods.length + 1);
        const maxFidelity = 100;
        
        methods.forEach((method, idx) => {
            const avgFidelity = method.data.quantumFidelity.reduce((a, b) => a + b, 0) / method.data.quantumFidelity.length * 100;
            const x = padding.left + (idx + 1) * barWidth;
            const barHeight = (avgFidelity / maxFidelity) * chartHeight;
            const y = height - padding.bottom - barHeight;
            
            // Gradient
            const gradient = ctx.createLinearGradient(x, y, x, height - padding.bottom);
            gradient.addColorStop(0, method.color);
            gradient.addColorStop(1, method.color + '80');
            ctx.fillStyle = gradient;
            ctx.fillRect(x - barWidth * 0.35, y, barWidth * 0.7, barHeight);
            
            // Labels
            ctx.fillStyle = '#e2e8f0';
            ctx.font = '11px Inter';
            ctx.textAlign = 'center';
            ctx.fillText(method.name.replace(' Quantum SLAM', ''), x, height - padding.bottom + 20);
            ctx.fillText(avgFidelity.toFixed(3) + '%', x, y - 5);
        });
        
        // Y-axis
        ctx.strokeStyle = '#6b7280';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.stroke();
        
        // Y-axis labels
        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px Inter';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const value = (i / 5) * maxFidelity;
            const y = height - padding.bottom - (i / 5) * chartHeight;
            ctx.fillText(value.toFixed(0) + '%', padding.left - 10, y + 4);
        }
        
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Average Quantum Fidelity (%)', width / 2, height - 20);
    }
    
    // Draw convergence comparison
    function drawConvergenceComparison() {
        const canvas = document.getElementById('quantum-convergence');
        if (!canvas) return;
        
        const container = canvas.parentElement;
        if (!container) return;
        const width = container.offsetWidth || 600;
        const height = 400;
        if (width === 0) return;
        
        canvas.width = width;
        canvas.height = height;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);
        
        const padding = { top: 40, right: 40, bottom: 60, left: 80 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        
        const methods = [
            { name: 'Quantum SLAM', data: quantumSLAMData.quantum, color: '#8b5cf6' },
            { name: 'Parallel Quantum SLAM', data: quantumSLAMData.parallel_quantum, color: '#ec4899' },
            { name: 'Concurrent Quantum SLAM', data: quantumSLAMData.concurrent_quantum, color: '#6366f1' }
        ];
        
        // Draw convergence curves (log scale)
        methods.forEach(method => {
            ctx.strokeStyle = method.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            const logConvergence = method.data.convergence.map(c => Math.log10(Math.max(c, 1e-12)));
            const minLog = Math.min(...logConvergence);
            const maxLog = Math.max(...logConvergence);
            const range = maxLog - minLog || 1;
            
            for (let i = 0; i < method.data.labels.length; i++) {
                const x = padding.left + (i / (method.data.labels.length - 1)) * chartWidth;
                const y = padding.top + chartHeight - ((logConvergence[i] - minLog) / range) * chartHeight;
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        });
        
        // Labels
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Step Size (h)', width / 2, height - 20);
        
        ctx.save();
        ctx.translate(20, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('log₁₀(Convergence Error)', 0, 0);
        ctx.restore();
        
        // Legend
        const legendY = padding.top + 10;
        methods.forEach((method, idx) => {
            ctx.fillStyle = method.color;
            ctx.fillRect(padding.left + 10, legendY + idx * 25, 20, 3);
            ctx.fillStyle = '#e2e8f0';
            ctx.font = '12px Inter';
            ctx.textAlign = 'left';
            ctx.fillText(method.name.replace(' Quantum SLAM', ''), padding.left + 35, legendY + idx * 25 + 8);
        });
    }
    
    // Draw accuracy-speed trade-off for quantum methods
    function drawQuantumAccuracySpeed() {
        const canvas = document.getElementById('quantum-accuracy-speed');
        if (!canvas) return;
        
        const container = canvas.parentElement;
        if (!container) return;
        const width = container.offsetWidth || 800;
        const height = 500;
        if (width === 0) return;
        
        canvas.width = width;
        canvas.height = height;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);
        
        const padding = 60;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        const methods = [
            { name: 'Quantum SLAM', data: quantumSLAMData.quantum, color: '#8b5cf6' },
            { name: 'Parallel Quantum SLAM', data: quantumSLAMData.parallel_quantum, color: '#ec4899' },
            { name: 'Concurrent Quantum SLAM', data: quantumSLAMData.concurrent_quantum, color: '#6366f1' }
        ];
        
        // Find ranges
        const allSpeeds = methods.flatMap(m => m.data.speed);
        const allAccuracies = methods.flatMap(m => m.data.accuracy);
        const minSpeed = Math.min(...allSpeeds);
        const maxSpeed = Math.max(...allSpeeds);
        const minAccuracy = Math.min(...allAccuracies);
        const maxAccuracy = Math.max(...allAccuracies);
        
        const speedRange = maxSpeed - minSpeed || 1;
        const accuracyRange = maxAccuracy - minAccuracy || 1;
        const speedScale = chartWidth / speedRange;
        const accuracyScale = chartHeight / accuracyRange;
        
        // Draw grid
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = padding + (chartHeight / 5) * i;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(padding + chartWidth, y);
            ctx.stroke();
        }
        
        // Draw data points
        methods.forEach(method => {
            const speeds = method.data.speed;
            const accuracies = method.data.accuracy;
            
            // Draw lines
            ctx.strokeStyle = method.color;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.3;
            ctx.beginPath();
            
            for (let i = 0; i < speeds.length; i++) {
                const x = padding + (speeds[i] - minSpeed) * speedScale;
                const y = padding + chartHeight - (accuracies[i] - minAccuracy) * accuracyScale;
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.globalAlpha = 1.0;
            
            // Draw points
            for (let i = 0; i < speeds.length; i++) {
                const x = padding + (speeds[i] - minSpeed) * speedScale;
                const y = padding + chartHeight - (accuracies[i] - minAccuracy) * accuracyScale;
                
                ctx.shadowColor = method.color;
                ctx.shadowBlur = 15;
                ctx.fillStyle = method.color;
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.fill();
                ctx.shadowBlur = 0;
            }
        });
        
        // Labels
        ctx.save();
        ctx.translate(20, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Accuracy (%)', 0, 0);
        ctx.restore();
        
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Speed (steps/sec)', width / 2, height - 20);
        
        // Legend
        const legendX = padding + chartWidth - 200;
        const legendY = padding + 20;
        methods.forEach((method, idx) => {
            const y = legendY + idx * 25;
            ctx.fillStyle = method.color;
            ctx.fillRect(legendX, y - 8, 15, 15);
            ctx.fillStyle = '#e2e8f0';
            ctx.font = '12px Inter';
            ctx.textAlign = 'left';
            ctx.fillText(method.name, legendX + 20, y + 4);
        });
    }
    
    // Initialize all quantum SLAM visualizations
    function initializeQuantumSLAM() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                setTimeout(initializeQuantumSLAM, 100);
            });
            return;
        }
        
        requestAnimationFrame(() => {
            setTimeout(() => {
                try {
                    drawQuantumStateEvolution();
                    drawQuantumFidelityComparison();
                    drawConvergenceComparison();
                    drawQuantumAccuracySpeed();
                } catch (e) {
                    console.error('[QuantumSLAM] Error initializing:', e);
                }
            }, 200);
        });
    }
    
    // Redraw on resize
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            drawQuantumStateEvolution();
            drawQuantumFidelityComparison();
            drawConvergenceComparison();
            drawQuantumAccuracySpeed();
        }, 250);
    });
    
    // Initialize
    initializeQuantumSLAM();
    
})();
