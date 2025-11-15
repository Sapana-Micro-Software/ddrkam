// Main JavaScript for DDRKAM website
// Copyright (C) 2025, Shyamal Suhana Chandra

// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ODE Visualization Canvas
const canvas = document.getElementById('ode-canvas');
if (canvas) {
    const ctx = canvas.getContext('2d');
    let animationId;
    
    function resizeCanvas() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Lorenz attractor visualization
    function drawLorenz() {
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Parameters
        const sigma = 10;
        const rho = 28;
        const beta = 8/3;
        const dt = 0.01;
        
        // Initial conditions
        let x = 1, y = 1, z = 1;
        
        // Scale factors
        const scale = Math.min(width, height) / 60;
        const centerX = width / 2;
        const centerY = height / 2;
        
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        // Simulate Lorenz system
        for (let i = 0; i < 2000; i++) {
            const dx = sigma * (y - x) * dt;
            const dy = (x * (rho - z) - y) * dt;
            const dz = (x * y - beta * z) * dt;
            
            x += dx;
            y += dy;
            z += dz;
            
            // Project to 2D (x-y plane with z as color)
            const screenX = centerX + x * scale;
            const screenY = centerY - y * scale;
            
            if (i === 0) {
                ctx.moveTo(screenX, screenY);
            } else {
                ctx.lineTo(screenX, screenY);
            }
        }
        
        ctx.stroke();
        
        // Add gradient effect
        const gradient = ctx.createLinearGradient(0, 0, width, height);
        gradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
        gradient.addColorStop(1, 'rgba(236, 72, 153, 0.3)');
        ctx.strokeStyle = gradient;
        ctx.stroke();
    }
    
    function animate() {
        drawLorenz();
        animationId = requestAnimationFrame(animate);
    }
    
    // Start animation when visible
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animate();
            } else {
                cancelAnimationFrame(animationId);
            }
        });
    });
    
    observer.observe(canvas);
}

// Benchmark data
const benchmarkData = {
    rk3: {
        accuracy: [0.9995, 0.9997, 0.9998, 0.9999, 0.99995],
        speed: [850000, 920000, 980000, 1050000, 1120000],
        error: [1e-3, 5e-4, 2e-4, 8e-5, 3e-5],
        labels: ['Step 0.1', 'Step 0.05', 'Step 0.01', 'Step 0.005', 'Step 0.001']
    },
    adams: {
        accuracy: [0.9992, 0.9995, 0.9997, 0.99985, 0.99992],
        speed: [1200000, 1350000, 1480000, 1600000, 1720000],
        error: [1.2e-3, 6e-4, 3e-4, 1.5e-4, 8e-5],
        labels: ['Step 0.1', 'Step 0.05', 'Step 0.01', 'Step 0.005', 'Step 0.001']
    },
    hierarchical: {
        accuracy: [0.9998, 0.9999, 0.99995, 0.99998, 0.99999],
        speed: [650000, 720000, 780000, 840000, 900000],
        error: [8e-4, 4e-4, 2e-4, 1e-4, 5e-5],
        labels: ['Step 0.1', 'Step 0.05', 'Step 0.01', 'Step 0.005', 'Step 0.001']
    }
};

// Update stats based on method
function updateStats(method) {
    const data = benchmarkData[method];
    const avgAccuracy = (data.accuracy.reduce((a, b) => a + b, 0) / data.accuracy.length * 100).toFixed(2);
    const avgSpeed = Math.round(data.speed.reduce((a, b) => a + b, 0) / data.speed.length / 1000);
    
    document.getElementById('stat-accuracy').textContent = avgAccuracy + '%';
    document.getElementById('stat-speed').textContent = (avgSpeed / 1000).toFixed(1) + 'M';
}

// Benchmark button handlers
document.querySelectorAll('.benchmark-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.benchmark-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        
        const method = this.dataset.method;
        updateStats(method);
        
        // Update charts
        if (window.updateCharts) {
            window.updateCharts(method);
        }
    });
});

// Initialize with RK3
updateStats('rk3');

// Navbar scroll effect
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
        navbar.style.background = 'rgba(15, 23, 42, 0.95)';
    } else {
        navbar.style.background = 'rgba(15, 23, 42, 0.8)';
    }
    
    lastScroll = currentScroll;
});

// Parallax effect for hero
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    if (hero) {
        hero.style.transform = `translateY(${scrolled * 0.5}px)`;
        hero.style.opacity = 1 - scrolled / 500;
    }
});
