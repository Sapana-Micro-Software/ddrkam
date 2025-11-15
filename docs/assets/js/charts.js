// Chart visualization for benchmarks
// Copyright (C) 2025, Shyamal Suhana Chandra

const benchmarkData = {
    rk3: {
        accuracy: [0.9995, 0.9997, 0.9998, 0.9999, 0.99995],
        speed: [850000, 920000, 980000, 1050000, 1120000],
        error: [1e-3, 5e-4, 2e-4, 8e-5, 3e-5],
        labels: ['0.1', '0.05', '0.01', '0.005', '0.001'],
        color: '#6366f1'
    },
    adams: {
        accuracy: [0.9992, 0.9995, 0.9997, 0.99985, 0.99992],
        speed: [1200000, 1350000, 1480000, 1600000, 1720000],
        error: [1.2e-3, 6e-4, 3e-4, 1.5e-4, 8e-5],
        labels: ['0.1', '0.05', '0.01', '0.005', '0.001'],
        color: '#8b5cf6'
    },
    hierarchical: {
        accuracy: [0.9998, 0.9999, 0.99995, 0.99998, 0.99999],
        speed: [650000, 720000, 780000, 840000, 900000],
        error: [8e-4, 4e-4, 2e-4, 1e-4, 5e-5],
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

function updateCharts(method) {
    const data = benchmarkData[method];
    
    // Accuracy chart (as percentage)
    const accuracyPercent = data.accuracy.map(a => a * 100);
    drawChart('accuracy-chart', accuracyPercent, data.labels, data.color, 'accuracy');
    
    // Speed chart
    drawChart('speed-chart', data.speed, data.labels, data.color, 'speed');
    
    // Error chart (log scale visualization)
    drawChart('error-chart', data.error, data.labels, data.color, 'error');
}

// Initialize charts on load
window.addEventListener('DOMContentLoaded', () => {
    updateCharts('rk3');
    
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
            }
        }, 250);
    });
});

// Update charts when benchmark button is clicked
document.querySelectorAll('.benchmark-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        updateCharts(this.dataset.method);
    });
});
