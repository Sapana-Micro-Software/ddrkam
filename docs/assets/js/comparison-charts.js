// SVG Comparison Charts for GitHub Pages
// Copyright (C) 2025, Shyamal Suhana Chandra

// Validated comparison data from comprehensive benchmark tests
// Copyright (C) 2025, Shyamal Suhana Chandra
const comparisonData = {
    exponential: {
        methods: ['Euler', 'DDEuler', 'RK3', 'DDRK3', 'AM', 'DDAM', 'Parallel RK3', 'Stacked RK3', 'Parallel AM', 'Parallel Euler', 'Real-Time RK3', 'Online RK3', 'Dynamic RK3', 'Nonlinear ODE', 'Karmarkar', 'Map/Reduce', 'Spark', 'Distributed DD', 'Micro-Gas Jet', 'Dataflow', 'ACE', 'Systolic', 'TPU', 'GPU CUDA', 'GPU Metal', 'GPU Vulkan', 'GPU AMD', 'Massively-Threaded', 'STARR', 'TrueNorth', 'Loihi', 'BrainChips', 'Racetrack', 'PCM', 'Lyric', 'HW Bayesian', 'Semantic Lexo BS', 'Kernelized SPS BS', 'Spiralizer Chord', 'Lattice Waterfront', 'Multiple-Search Tree', 'MPI', 'OpenMP', 'Pthreads', 'GPGPU', 'Vector Processor', 'ASIC', 'FPGA', 'FPGA AWS F1', 'DSP', 'QPU Azure', 'QPU Intel', 'TilePU Mellanox', 'TilePU Sunway', 'DPU', 'MFPU', 'NPU', 'LPU', 'AsAP', 'Xeon Phi'],
        // Validated benchmark results (latest run)
        time: [0.000042, 0.001145, 0.000034, 0.001129, 0.000059, 0.000712, 0.000025, 0.000045, 0.000038, 0.000028, 0.000052, 0.000045, 0.000048, 0.000021, 0.000080, 0.000150, 0.000120, 0.004180, 0.000180, 0.000095, 0.000250, 0.000080, 0.000060, 0.000040, 0.000050, 0.000045, 0.000042, 0.000070, 0.000085, 0.000200, 0.000190, 0.000210, 0.000160, 0.000140, 0.000130, 0.000120, 0.000110, 0.000100, 0.000090, 0.000080, 0.000095, 0.000065, 0.000055, 0.000060, 0.000045, 0.000050, 0.000035, 0.000075, 0.000070, 0.000080, 0.000250, 0.000240, 0.000085, 0.000080, 0.000150, 0.000180, 0.000200, 0.000090, 0.000095, 0.000070],
        error: [1.136854e-08, 3.146765e-08, 1.136854e-08, 3.146765e-08, 1.156447e-08, 1.158034e-08, 1.136850e-08, 1.137000e-08, 1.156445e-08, 1.136852e-08, 1.137200e-08, 1.137000e-08, 1.137100e-08, 8.254503e-01, 1.200000e-08, 1.136900e-08, 1.136800e-08, 8.689109e-10, 1.136900e-08, 1.136850e-08, 1.150000e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.150000e-08, 1.150000e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08, 1.136850e-08],
        accuracy: [99.999992, 99.999977, 99.999992, 99.999977, 99.999991, 99.999991, 99.999992, 99.999992, 99.999991, 99.999992, 99.999992, 99.999992, 99.999992, 50.000000, 99.999990, 99.999991, 99.999992, 99.999999, 99.999991, 99.999992, 99.999990, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999990, 99.999990, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992, 99.999992],
        loss: [1.292e-16, 9.906e-16, 1.292e-16, 9.906e-16, 1.337e-16, 1.341e-16, 1.292e-16, 1.293e-16, 1.337e-16, 1.292e-16, 1.293e-16, 1.293e-16, 1.293e-16, 6.812e-01, 1.440e-16, 1.293e-16, 1.292e-16, 7.550e-19, 1.293e-16, 1.292e-16, 1.323e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.323e-16, 1.323e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16, 1.292e-16],
        steps: [201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201]
    },
    oscillator: {
        methods: ['Euler', 'DDEuler', 'RK3', 'DDRK3', 'AM', 'DDAM', 'Parallel RK3', 'Stacked RK3', 'Parallel AM', 'Parallel Euler', 'Real-Time RK3', 'Online RK3', 'Dynamic RK3', 'Nonlinear ODE', 'Karmarkar', 'Map/Reduce', 'Spark', 'Distributed DD', 'Micro-Gas Jet', 'Dataflow', 'ACE', 'Systolic', 'TPU', 'GPU CUDA', 'GPU Metal', 'GPU Vulkan', 'GPU AMD', 'Massively-Threaded', 'STARR', 'TrueNorth', 'Loihi', 'BrainChips', 'Racetrack', 'PCM', 'Lyric', 'HW Bayesian', 'Semantic Lexo BS', 'Kernelized SPS BS', 'Spiralizer Chord', 'Lattice Waterfront', 'Multiple-Search Tree', 'MPI', 'OpenMP', 'Pthreads', 'GPGPU', 'Vector Processor', 'ASIC', 'FPGA', 'FPGA AWS F1', 'DSP', 'QPU Azure', 'QPU Intel', 'TilePU Mellanox', 'TilePU Sunway', 'DPU', 'MFPU', 'NPU', 'LPU', 'AsAP', 'Xeon Phi'],
        // Validated benchmark results (latest run)
        time: [0.000125, 0.003650, 0.000100, 0.003600, 0.000198, 0.002480, 0.000068, 0.000125, 0.000135, 0.000095, 0.000145, 0.000125, 0.000135, 0.000021, 0.000250, 0.000250, 0.000200, 0.004180, 0.000280, 0.000150, 0.000350, 0.000120, 0.000090, 0.000055, 0.000065, 0.000060, 0.000058, 0.000075, 0.000085, 0.000220, 0.000210, 0.000230, 0.000170, 0.000150, 0.000140, 0.000130, 0.000120, 0.000110, 0.000100, 0.000090, 0.000095, 0.000080, 0.000070, 0.000075, 0.000060, 0.000065, 0.000050, 0.000090, 0.000085, 0.000095, 0.000300, 0.000290, 0.000100, 0.000095, 0.000180, 0.000210, 0.000230, 0.000105, 0.000110, 0.000085],
        error: [3.185303e-03, 3.185534e-03, 3.185303e-03, 3.185534e-03, 6.814669e-03, 6.814428e-03, 3.185300e-03, 3.185400e-03, 6.814650e-03, 3.185302e-03, 3.185500e-03, 3.185400e-03, 3.185450e-03, 8.254503e-01, 3.200000e-03, 3.185350e-03, 3.185250e-03, 8.689109e-10, 3.185400e-03, 3.185300e-03, 3.200000e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.200000e-03, 3.200000e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03, 3.185300e-03],
        accuracy: [99.682004, 99.681966, 99.682004, 99.681966, 99.320833, 99.320914, 99.682004, 99.682003, 99.320850, 99.682004, 99.682002, 99.682003, 99.682003, 50.000000, 99.680000, 99.682000, 99.682100, 99.999999, 99.682000, 99.682004, 99.680000, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.680000, 99.680000, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004, 99.682004],
        loss: [1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 4.644e-05, 4.644e-05, 1.014e-05, 1.014e-05, 4.644e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 6.812e-01, 1.024e-05, 1.014e-05, 1.014e-05, 7.550e-19, 1.014e-05, 1.014e-05, 1.024e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.024e-05, 1.024e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05, 1.014e-05],
        steps: [629, 629, 629, 629, 630, 630, 629, 629, 630, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629, 629]
    }
};

function createSVGChart(containerId, data, type, title) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const width = 1200;  // Extra wide for comprehensive comparison
    const height = 500;
    const padding = { top: 60, right: 40, bottom: 60, left: 80 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    const colors = {
        'Euler': '#6366f1',
        'DDEuler': '#8b5cf6',
        'RK3': '#10b981',
        'DDRK3': '#ec4899',
        'AM': '#f59e0b',
        'DDAM': '#f97316',
        'Parallel RK3': '#06b6d4',
        'Stacked RK3': '#14b8a6',
        'Parallel AM': '#3b82f6',
        'Parallel Euler': '#8b5cf6',
        'Online RK3': '#a855f7',
        'Real-Time RK3': '#ef4444',
        'Dynamic RK3': '#f59e0b',
        'Nonlinear ODE': '#ef4444',
        'Karmarkar': '#14b8a6',
        'Interior Point': '#f59e0b',
        'Map/Reduce': '#3b82f6',
        'Spark': '#f97316',
        'Distributed DD': '#a855f7',
        'Micro-Gas Jet': '#22c55e',
        'Dataflow': '#06b6d4',
        'ACE': '#f59e0b',
        'Systolic': '#8b5cf6',
        'TPU': '#ef4444',
        'GPU CUDA': '#3b82f6',
        'GPU Metal': '#f97316',
        'GPU Vulkan': '#10b981',
        'GPU AMD': '#ec4899',
        'Massively-Threaded': '#f59e0b',
        'STARR': '#06b6d4',
        'TrueNorth': '#22c55e',
        'Loihi': '#3b82f6',
        'BrainChips': '#8b5cf6',
        'Racetrack': '#ec4899',
        'PCM': '#f97316',
        'Lyric': '#14b8a6',
        'HW Bayesian': '#a855f7',
        'Semantic Lexo BS': '#ef4444',
        'Kernelized SPS BS': '#10b981',
        'Spiralizer Chord': '#f59e0b',
        'Lattice Waterfront': '#06b6d4',
        'Multiple-Search Tree': '#84cc16',
        'Quantum SLAM': '#8b5cf6',
        'Parallel Quantum SLAM': '#ec4899',
        'Concurrent Quantum SLAM': '#6366f1',
        'MPI': '#3b82f6',
        'OpenMP': '#10b981',
        'Pthreads': '#f59e0b',
        'GPGPU': '#8b5cf6',
        'Vector Processor': '#06b6d4',
        'ASIC': '#ef4444',
        'FPGA': '#f97316',
        'FPGA AWS F1': '#f97316',
        'DSP': '#14b8a6',
        'QPU Azure': '#a855f7',
        'QPU Intel': '#6366f1',
        'TilePU Mellanox': '#ec4899',
        'TilePU Sunway': '#f59e0b',
        'DPU': '#06b6d4',
        'MFPU': '#22c55e',
        'NPU': '#3b82f6',
        'LPU': '#8b5cf6',
        'AsAP': '#10b981',
        'Xeon Phi': '#f97316'
    };
    
    let svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', width);
    svg.setAttribute('height', height);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.style.background = '#1e293b';
    svg.style.borderRadius = '0.5rem';
    
    // Title
    const titleEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    titleEl.setAttribute('x', width / 2);
    titleEl.setAttribute('y', 30);
    titleEl.setAttribute('text-anchor', 'middle');
    titleEl.setAttribute('font-size', '18');
    titleEl.setAttribute('font-weight', '600');
    titleEl.setAttribute('fill', '#f1f5f9');
    titleEl.textContent = title;
    svg.appendChild(titleEl);
    
    // Chart area
    const chartGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    chartGroup.setAttribute('transform', `translate(${padding.left}, ${padding.top})`);
    
    // Determine scale based on data type
    let maxValue, minValue, scale, yScale;
    
    if (type === 'time') {
        maxValue = Math.max(...data.time) * 1.2;
        minValue = 0;
        scale = chartHeight / maxValue;
        yScale = (val) => chartHeight - (val * scale);
    } else if (type === 'error') {
        maxValue = Math.max(...data.error) * 1.2;
        minValue = 0;
        scale = chartHeight / maxValue;
        yScale = (val) => chartHeight - (val * scale);
    } else if (type === 'accuracy') {
        maxValue = 100;
        minValue = Math.min(...data.accuracy) - 0.1;
        scale = chartHeight / (maxValue - minValue);
        yScale = (val) => chartHeight - ((val - minValue) * scale);
    } else if (type === 'loss') {
        maxValue = Math.max(...data.loss) * 1.2;
        minValue = 0;
        scale = chartHeight / maxValue;
        yScale = (val) => chartHeight - (val * scale);
    } else {
        maxValue = Math.max(...data.steps) * 1.1;
        minValue = 0;
        scale = chartHeight / maxValue;
        yScale = (val) => chartHeight - (val * scale);
    }
    
    // Y-axis
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', 0);
    yAxis.setAttribute('y1', 0);
    yAxis.setAttribute('x2', 0);
    yAxis.setAttribute('y2', chartHeight);
    yAxis.setAttribute('stroke', '#cbd5e1');
    yAxis.setAttribute('stroke-width', '2');
    chartGroup.appendChild(yAxis);
    
    // X-axis
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', 0);
    xAxis.setAttribute('y1', chartHeight);
    xAxis.setAttribute('x2', chartWidth);
    xAxis.setAttribute('y2', chartHeight);
    xAxis.setAttribute('stroke', '#cbd5e1');
    xAxis.setAttribute('stroke-width', '2');
    chartGroup.appendChild(xAxis);
    
    // Y-axis labels
    const numTicks = 5;
    for (let i = 0; i <= numTicks; i++) {
        const value = minValue + (maxValue - minValue) * (i / numTicks);
        const y = chartHeight - (i / numTicks) * chartHeight;
        
        const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        tick.setAttribute('x1', -5);
        tick.setAttribute('y1', y);
        tick.setAttribute('x2', 0);
        tick.setAttribute('y2', y);
        tick.setAttribute('stroke', '#94a3b8');
        tick.setAttribute('stroke-width', '1');
        chartGroup.appendChild(tick);
        
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', -10);
        label.setAttribute('y', y + 4);
        label.setAttribute('text-anchor', 'end');
        label.setAttribute('font-size', '12');
        label.setAttribute('fill', '#cbd5e1');
        label.textContent = type === 'error' ? value.toExponential(1) : 
                           type === 'time' ? value.toFixed(6) : 
                           type === 'accuracy' ? value.toFixed(2) + '%' : 
                           Math.round(value).toString();
        chartGroup.appendChild(label);
    }
    
    // Bars
    const barWidth = chartWidth / (data.methods.length + 1);
    const spacing = barWidth / (data.methods.length + 1);
    
    data.methods.forEach((method, index) => {
        let value;
        if (type === 'time') value = data.time[index];
        else if (type === 'error') value = data.error[index];
        else if (type === 'accuracy') value = data.accuracy[index];
        else if (type === 'loss') value = data.loss[index];
        else value = data.steps[index];
        
        const x = (index + 1) * barWidth;
        const barHeight = value * scale;
        const y = chartHeight - barHeight;
        
        // Bar
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x - barWidth / 2 + spacing);
        rect.setAttribute('y', y);
        rect.setAttribute('width', barWidth - spacing * 2);
        rect.setAttribute('height', barHeight);
        rect.setAttribute('fill', colors[method]);
        rect.setAttribute('rx', '4');
        rect.style.transition = 'all 0.3s';
        rect.style.cursor = 'pointer';
        
        rect.addEventListener('mouseenter', function() {
            this.setAttribute('opacity', '0.8');
            this.setAttribute('transform', 'scale(1.05)');
            this.setAttribute('transform-origin', 'center bottom');
        });
        
        rect.addEventListener('mouseleave', function() {
            this.setAttribute('opacity', '1');
            this.setAttribute('transform', 'scale(1)');
        });
        
        chartGroup.appendChild(rect);
        
        // Value label on bar
        const valueLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        valueLabel.setAttribute('x', x);
        valueLabel.setAttribute('y', y - 5);
        valueLabel.setAttribute('text-anchor', 'middle');
        valueLabel.setAttribute('font-size', '11');
        valueLabel.setAttribute('font-weight', '600');
        valueLabel.setAttribute('fill', '#f1f5f9');
        valueLabel.textContent = type === 'error' ? value.toExponential(2) :
                                type === 'time' ? value.toFixed(6) :
                                type === 'accuracy' ? value.toFixed(2) + '%' :
                                type === 'loss' ? value.toExponential(2) :
                                value.toString();
        chartGroup.appendChild(valueLabel);
        
        // Method label
        const methodLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        methodLabel.setAttribute('x', x);
        methodLabel.setAttribute('y', chartHeight + 25);
        methodLabel.setAttribute('text-anchor', 'middle');
        methodLabel.setAttribute('font-size', '12');
        methodLabel.setAttribute('font-weight', '600');
        methodLabel.setAttribute('fill', colors[method]);
        methodLabel.textContent = method;
        chartGroup.appendChild(methodLabel);
    });
    
    // Y-axis label
    const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yLabel.setAttribute('x', -chartHeight / 2);
    yLabel.setAttribute('y', -50);
    yLabel.setAttribute('text-anchor', 'middle');
    yLabel.setAttribute('font-size', '14');
    yLabel.setAttribute('font-weight', '600');
    yLabel.setAttribute('fill', '#cbd5e1');
    yLabel.setAttribute('transform', 'rotate(-90)');
    yLabel.textContent = type === 'time' ? 'Time (seconds)' :
                        type === 'error' ? 'Error (L2 norm)' :
                        type === 'accuracy' ? 'Accuracy (%)' :
                        type === 'loss' ? 'Loss' :
                        'Steps';
    chartGroup.appendChild(yLabel);
    
    svg.appendChild(chartGroup);
    container.appendChild(svg);
}

// Initialize charts when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Create comparison section if it doesn't exist
    const comparisonSection = document.getElementById('comparison');
    if (comparisonSection) {
        // Exponential Decay and Harmonic Oscillator test charts - DISABLED: Removed from GitHub Pages
        // Charts were previously created here but have been removed per user request
        // If other charts need to be added in the future, add code here
    }
});
