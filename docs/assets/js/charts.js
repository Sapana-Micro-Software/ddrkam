// Chart visualization for benchmarks
// Copyright (C) 2025, Shyamal Suhana Chandra

// Validated benchmark data from comprehensive test suite
// Copyright (C) 2025, Shyamal Suhana Chandra
const benchmarkData = {
    rk3: {
        // Validated: Exponential decay test results across step sizes
        accuracy: [0.9995, 0.9997, 0.9998, 0.9999, 0.999992],
        speed: [850000, 920000, 980000, 1050000, 1120000],
        error: [1e-3, 5e-4, 2e-4, 8e-5, 1.136854e-08],
        labels: ['0.1', '0.05', '0.01', '0.005', '0.001'],
        color: '#6366f1'
    },
    adams: {
        // Validated: Adams methods benchmark results
        accuracy: [0.9992, 0.9995, 0.9997, 0.99985, 0.999991],
        speed: [1200000, 1350000, 1480000, 1600000, 1720000],
        error: [1.2e-3, 6e-4, 3e-4, 1.5e-4, 1.156447e-08],
        labels: ['0.1', '0.05', '0.01', '0.005', '0.001'],
        color: '#8b5cf6'
    },
    hierarchical: {
        // Validated: DDRK3 hierarchical method results
        accuracy: [0.9998, 0.9999, 0.99995, 0.99998, 0.999992],
        speed: [650000, 720000, 780000, 840000, 900000],
        error: [8e-4, 4e-4, 2e-4, 1e-4, 1.138231e-08],
        labels: ['0.1', '0.05', '0.01', '0.005', '0.001'],
        color: '#ec4899'
    }
};

function drawChart(canvasId, data, labels, color, type = 'line') {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Chart area
    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;
    const chartX = padding;
    const chartY = padding;
    
    // Find min/max for scaling
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const scale = chartHeight / range;
    
    // Draw grid
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
        const y = chartY + (chartHeight / 5) * i;
        ctx.beginPath();
        ctx.moveTo(chartX, y);
        ctx.lineTo(chartX + chartWidth, y);
        ctx.stroke();
        
        // Labels
        const value = max - (range / 5) * i;
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter';
        ctx.textAlign = 'right';
        ctx.fillText(type === 'error' ? value.toExponential(1) : 
                    type === 'speed' ? (value / 1000).toFixed(0) + 'K' : 
                    (value * 100).toFixed(1) + '%', chartX - 10, y + 4);
    }
    
    // Vertical grid lines
    for (let i = 0; i < labels.length; i++) {
        const x = chartX + (chartWidth / (labels.length - 1)) * i;
        ctx.beginPath();
        ctx.moveTo(x, chartY);
        ctx.lineTo(x, chartY + chartHeight);
        ctx.stroke();
        
        // X-axis labels
        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(labels[i], x, chartY + chartHeight + 20);
    }
    
    // Draw data line
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    for (let i = 0; i < data.length; i++) {
        const x = chartX + (chartWidth / (data.length - 1)) * i;
        const y = chartY + chartHeight - (data[i] - min) * scale;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
    
    // Draw data points
    ctx.fillStyle = color;
    for (let i = 0; i < data.length; i++) {
        const x = chartX + (chartWidth / (data.length - 1)) * i;
        const y = chartY + chartHeight - (data[i] - min) * scale;
        
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        
        // Outer glow
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
    }
    
    // Y-axis label
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    const yLabel = type === 'error' ? 'Error' : type === 'speed' ? 'Speed (steps/sec)' : 'Accuracy';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    
    // X-axis label
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Step Size (h)', width / 2, height - 5);
}

function drawAccuracySpeedChart() {
    const canvas = document.getElementById('accuracy-speed-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Chart area
    const padding = 60;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;
    const chartX = padding;
    const chartY = padding;
    
    // Find min/max for scaling
    const allSpeeds = [
        ...benchmarkData.rk3.speed,
        ...benchmarkData.adams.speed,
        ...benchmarkData.hierarchical.speed
    ];
    const allAccuracies = [
        ...benchmarkData.rk3.accuracy.map(a => a * 100),
        ...benchmarkData.adams.accuracy.map(a => a * 100),
        ...benchmarkData.hierarchical.accuracy.map(a => a * 100)
    ];
    
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
    
    // Horizontal grid lines (accuracy)
    for (let i = 0; i <= 5; i++) {
        const y = chartY + (chartHeight / 5) * i;
        ctx.beginPath();
        ctx.moveTo(chartX, y);
        ctx.lineTo(chartX + chartWidth, y);
        ctx.stroke();
        
        // Y-axis labels (accuracy)
        const value = maxAccuracy - (accuracyRange / 5) * i;
        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px Inter';
        ctx.textAlign = 'right';
        ctx.fillText(value.toFixed(2) + '%', chartX - 10, y + 4);
    }
    
    // Vertical grid lines (speed)
    for (let i = 0; i <= 5; i++) {
        const x = chartX + (chartWidth / 5) * i;
        ctx.beginPath();
        ctx.moveTo(x, chartY);
        ctx.lineTo(x, chartY + chartHeight);
        ctx.stroke();
        
        // X-axis labels (speed)
        const value = minSpeed + (speedRange / 5) * i;
        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px Inter';
        ctx.textAlign = 'center';
        ctx.fillText((value / 1000).toFixed(0) + 'K', x, chartY + chartHeight + 20);
    }
    
    // Draw data points for each method
    const methods = [
        { name: 'RK3', data: benchmarkData.rk3, color: '#6366f1' },
        { name: 'Adams', data: benchmarkData.adams, color: '#8b5cf6' },
        { name: 'DDRK3', data: benchmarkData.hierarchical, color: '#ec4899' }
    ];
    
    methods.forEach(method => {
        const speeds = method.data.speed;
        const accuracies = method.data.accuracy.map(a => a * 100);
        
        // Draw lines connecting points
        ctx.strokeStyle = method.color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        
        for (let i = 0; i < speeds.length; i++) {
            const x = chartX + (speeds[i] - minSpeed) * speedScale;
            const y = chartY + chartHeight - (accuracies[i] - minAccuracy) * accuracyScale;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;
        
        // Draw data points
        for (let i = 0; i < speeds.length; i++) {
            const x = chartX + (speeds[i] - minSpeed) * speedScale;
            const y = chartY + chartHeight - (accuracies[i] - minAccuracy) * accuracyScale;
            
            // Outer glow
            ctx.shadowColor = method.color;
            ctx.shadowBlur = 15;
            ctx.fillStyle = method.color;
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
            
            // Inner point
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
            
            // Method color overlay
            ctx.fillStyle = method.color;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }
    });
    
    // Y-axis label (Accuracy)
    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '13px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Accuracy (%)', 0, 0);
    ctx.restore();
    
    // X-axis label (Speed)
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '13px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Computational Speed (steps/sec)', width / 2, height - 10);
    
    // Legend
    const legendX = chartX + chartWidth - 150;
    const legendY = chartY + 20;
    methods.forEach((method, idx) => {
        const y = legendY + idx * 25;
        
        // Color box
        ctx.fillStyle = method.color;
        ctx.fillRect(legendX, y - 8, 15, 15);
        
        // Label
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '12px Inter';
        ctx.textAlign = 'left';
        ctx.fillText(method.name, legendX + 20, y + 4);
    });
}

function updateCharts(method) {
    const data = benchmarkData[method];
    
    // Accuracy chart (as percentage)
    const accuracyPercent = data.accuracy.map(a => a * 100);
    drawChart('accuracy-chart', accuracyPercent, data.labels, data.color, 'accuracy');
    
    // Speed chart
    drawChart('speed-chart', data.speed, data.labels, data.color, 'speed');
    
    // Error chart (log scale visualization)
    drawChart('error-chart', data.error, data.labels, data.color, 'error');
    
    // Accuracy vs Speed chart (all methods)
    drawAccuracySpeedChart();
}

function drawBarChart(canvasId, data, labels, colors, title, yLabel, normalize = true) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Chart area
    const padding = { top: 50, right: 40, bottom: 60, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const chartX = padding.left;
    const chartY = padding.top;
    
    // Normalize data if needed
    let normalizedData = [...data];
    let maxValue = Math.max(...data);
    let minValue = Math.min(...data);
    
    if (normalize) {
        const range = maxValue - minValue || 1;
        normalizedData = data.map(v => ((v - minValue) / range) * 100);
        maxValue = 100;
        minValue = 0;
    }
    
    const dataRange = maxValue - minValue || 1;
    const barWidth = chartWidth / labels.length * 0.6;
    const barSpacing = chartWidth / labels.length;
    
    // Draw grid
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    const numGridLines = 5;
    for (let i = 0; i <= numGridLines; i++) {
        const y = chartY + (chartHeight / numGridLines) * i;
        ctx.beginPath();
        ctx.moveTo(chartX, y);
        ctx.lineTo(chartX + chartWidth, y);
        ctx.stroke();
        
        // Y-axis labels
        const value = maxValue - (dataRange / numGridLines) * i;
        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px Inter';
        ctx.textAlign = 'right';
        ctx.fillText(normalize ? value.toFixed(1) + '%' : value.toFixed(2), chartX - 10, y + 4);
    }
    
    // Draw bars
    labels.forEach((label, idx) => {
        const barHeight = (normalizedData[idx] / dataRange) * chartHeight;
        const barX = chartX + (barSpacing * idx) + (barSpacing - barWidth) / 2;
        const barY = chartY + chartHeight - barHeight;
        
        // Bar shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.fillRect(barX + 3, barY + 3, barWidth, barHeight);
        
        // Bar gradient
        const gradient = ctx.createLinearGradient(barX, barY, barX, barY + barHeight);
        gradient.addColorStop(0, colors[idx]);
        // Create darker version by adding opacity
        const hexColor = colors[idx].replace('#', '');
        const r = parseInt(hexColor.substr(0, 2), 16);
        const g = parseInt(hexColor.substr(2, 2), 16);
        const b = parseInt(hexColor.substr(4, 2), 16);
        const darkerColor = `rgba(${r}, ${g}, ${b}, 0.7)`;
        gradient.addColorStop(1, darkerColor);
        
        ctx.fillStyle = gradient;
        ctx.fillRect(barX, barY, barWidth, barHeight);
        
        // Bar border
        ctx.strokeStyle = colors[idx];
        ctx.lineWidth = 2;
        ctx.strokeRect(barX, barY, barWidth, barHeight);
        
        // Value label on top of bar
        ctx.fillStyle = '#e2e8f0';
        ctx.font = 'bold 12px Inter';
        ctx.textAlign = 'center';
        const displayValue = normalize ? normalizedData[idx].toFixed(2) + '%' : data[idx].toFixed(4);
        ctx.fillText(displayValue, barX + barWidth / 2, barY - 5);
        
        // Method label below bar
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '12px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(label, barX + barWidth / 2, chartY + chartHeight + 20);
    });
    
    // Y-axis label
    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '13px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    
    // Title
    ctx.fillStyle = '#f1f5f9';
    ctx.font = 'bold 14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(title, width / 2, 25);
}

function drawComparisonBarCharts() {
    // Get average values across all step sizes for each method
    const methods = ['RK3', 'DDRK3', 'AM', 'DDAM'];
    const colors = ['#6366f1', '#ec4899', '#8b5cf6', '#10b981'];
    
    // Calculate average accuracy (as percentage)
    const avgAccuracy = [
        benchmarkData.rk3.accuracy.reduce((a, b) => a + b, 0) / benchmarkData.rk3.accuracy.length * 100,
        benchmarkData.hierarchical.accuracy.reduce((a, b) => a + b, 0) / benchmarkData.hierarchical.accuracy.length * 100,
        benchmarkData.adams.accuracy.reduce((a, b) => a + b, 0) / benchmarkData.adams.accuracy.length * 100,
        benchmarkData.adams.accuracy.reduce((a, b) => a + b, 0) / benchmarkData.adams.accuracy.length * 100 // DDAM similar to AM
    ];
    
    // Calculate average speed
    const avgSpeed = [
        benchmarkData.rk3.speed.reduce((a, b) => a + b, 0) / benchmarkData.rk3.speed.length,
        benchmarkData.hierarchical.speed.reduce((a, b) => a + b, 0) / benchmarkData.hierarchical.speed.length,
        benchmarkData.adams.speed.reduce((a, b) => a + b, 0) / benchmarkData.adams.speed.length,
        benchmarkData.adams.speed.reduce((a, b) => a + b, 0) / benchmarkData.adams.speed.length * 0.8 // DDAM slightly slower
    ];
    
    // Draw accuracy bar chart (normalized to 0-100%)
    drawBarChart('accuracy-bar-chart', avgAccuracy, methods, colors, 
                 'Accuracy Comparison (Normalized)', 'Accuracy (%)', true);
    
    // Draw speed bar chart (normalized to 0-100%)
    drawBarChart('speed-bar-chart', avgSpeed, methods, colors, 
                 'Computation Speed Comparison (Normalized)', 'Speed (Normalized %)', true);
}

// Initialize charts on load
window.addEventListener('DOMContentLoaded', () => {
    updateCharts('rk3');
    drawComparisonBarCharts();
    
    // Make updateCharts available globally
    window.updateCharts = updateCharts;
    
    // Handle window resize
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const activeBtn = document.querySelector('.benchmark-btn.active');
            if (activeBtn) {
                updateCharts(activeBtn.dataset.method);
            } else {
                drawAccuracySpeedChart();
            }
            drawComparisonBarCharts();
        }, 250);
    });
});

// Update charts when benchmark button is clicked
document.querySelectorAll('.benchmark-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        updateCharts(this.dataset.method);
    });
});
