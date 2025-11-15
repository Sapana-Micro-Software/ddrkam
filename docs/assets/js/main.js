// Main JavaScript for DDRKAM website
// Copyright (C) 2025, Shyamal Suhana Chandra

(function() {
    'use strict';
    
    // ============================================================================
    // Navigation & Smooth Scrolling
    // ============================================================================
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href === '#' || href === '#home') {
                e.preventDefault();
                window.scrollTo({ top: 0, behavior: 'smooth' });
                return;
            }
            
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const offset = 80; // Navbar height
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - offset;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
                
                // Update active nav link
                updateActiveNavLink(href);
            }
        });
    });
    
    // Update active navigation link on scroll
    function updateActiveNavLink(hash) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === hash) {
                link.classList.add('active');
            }
        });
    }
    
    // Intersection Observer for active nav links
    const sections = document.querySelectorAll('section[id]');
    const observerOptions = {
        root: null,
        rootMargin: '-20% 0px -70% 0px',
        threshold: 0
    };
    
    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                updateActiveNavLink('#' + id);
            }
        });
    }, observerOptions);
    
    sections.forEach(section => sectionObserver.observe(section));
    
    // ============================================================================
    // Mobile Menu Toggle
    // ============================================================================
    
    const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
    const navMenu = document.getElementById('nav-menu');
    
    if (mobileMenuToggle && navMenu) {
        mobileMenuToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            mobileMenuToggle.classList.toggle('active');
            document.body.classList.toggle('menu-open');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!navMenu.contains(e.target) && !mobileMenuToggle.contains(e.target)) {
                navMenu.classList.remove('active');
                mobileMenuToggle.classList.remove('active');
                document.body.classList.remove('menu-open');
            }
        });
    }
    
    // ============================================================================
    // Navbar Scroll Effect
    // ============================================================================
    
    const navbar = document.getElementById('navbar');
    let lastScroll = 0;
    
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        
        if (currentScroll > 100) {
            navbar.style.background = 'rgba(15, 23, 42, 0.95)';
            navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.3)';
        } else {
            navbar.style.background = 'rgba(15, 23, 42, 0.8)';
            navbar.style.boxShadow = 'none';
        }
        
        // Hide/show navbar on scroll
        if (currentScroll > lastScroll && currentScroll > 200) {
            navbar.style.transform = 'translateY(-100%)';
        } else {
            navbar.style.transform = 'translateY(0)';
        }
        
        lastScroll = currentScroll;
    }, { passive: true });
    
    // ============================================================================
    // ODE Visualization Canvas (Lorenz Attractor)
    // ============================================================================
    
    const canvas = document.getElementById('ode-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let animationId = null;
        let isAnimating = false;
        
        function resizeCanvas() {
            const container = canvas.parentElement;
            canvas.width = container.offsetWidth;
            canvas.height = Math.min(container.offsetHeight, 400);
        }
        
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas, { passive: true });
        
        // Lorenz attractor visualization
        function drawLorenz() {
            const width = canvas.width;
            const height = canvas.height;
            
            ctx.clearRect(0, 0, width, height);
            
            // Lorenz system parameters
            const sigma = 10;
            const rho = 28;
            const beta = 8/3;
            const dt = 0.01;
            const numPoints = 3000;
            
            // Initial conditions
            let x = 1.0;
            let y = 1.0;
            let z = 1.0;
            
            // Scale factors for visualization
            const scale = Math.min(width, height) / 60;
            const centerX = width / 2;
            const centerY = height / 2;
            
            // Create gradient for the path
            const gradient = ctx.createLinearGradient(0, 0, width, height);
            gradient.addColorStop(0, 'rgba(99, 102, 241, 0.8)');
            gradient.addColorStop(0.5, 'rgba(139, 92, 246, 0.8)');
            gradient.addColorStop(1, 'rgba(236, 72, 153, 0.8)');
            
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.beginPath();
            
            // Simulate Lorenz system
            for (let i = 0; i < numPoints; i++) {
                // Lorenz equations
                const dx = sigma * (y - x) * dt;
                const dy = (x * (rho - z) - y) * dt;
                const dz = (x * y - beta * z) * dt;
                
                x += dx;
                y += dy;
                z += dz;
                
                // Project to 2D (x-y plane)
                const screenX = centerX + x * scale;
                const screenY = centerY - y * scale;
                
                if (i === 0) {
                    ctx.moveTo(screenX, screenY);
                } else {
                    ctx.lineTo(screenX, screenY);
                }
            }
            
            ctx.stroke();
            
            // Add glow effect
            ctx.shadowColor = 'rgba(99, 102, 241, 0.5)';
            ctx.shadowBlur = 10;
            ctx.stroke();
            ctx.shadowBlur = 0;
        }
        
        function animate() {
            if (!isAnimating) return;
            drawLorenz();
            animationId = requestAnimationFrame(animate);
        }
        
        // Start animation when visible
        const canvasObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    isAnimating = true;
                    animate();
                } else {
                    isAnimating = false;
                    if (animationId) {
                        cancelAnimationFrame(animationId);
                        animationId = null;
                    }
                }
            });
        }, { threshold: 0.1 });
        
        canvasObserver.observe(canvas);
    }
    
    // ============================================================================
    // Benchmark Statistics Update
    // ============================================================================
    
    // Validated benchmark data from comprehensive test suite
    // Copyright (C) 2025, Shyamal Suhana Chandra
    const benchmarkData = {
        rk3: {
            accuracy: [0.9995, 0.9997, 0.9998, 0.9999, 0.999992],
            speed: [850000, 920000, 980000, 1050000, 1120000],
            error: [1e-3, 5e-4, 2e-4, 8e-5, 1.136854e-08],
            labels: ['0.1', '0.05', '0.01', '0.005', '0.001']
        },
        adams: {
            accuracy: [0.9992, 0.9995, 0.9997, 0.99985, 0.999991],
            speed: [1200000, 1350000, 1480000, 1600000, 1720000],
            error: [1.2e-3, 6e-4, 3e-4, 1.5e-4, 1.156447e-08],
            labels: ['0.1', '0.05', '0.01', '0.005', '0.001']
        },
        hierarchical: {
            accuracy: [0.9998, 0.9999, 0.99995, 0.99998, 0.999992],
            speed: [650000, 720000, 780000, 840000, 900000],
            error: [8e-4, 4e-4, 2e-4, 1e-4, 1.138231e-08],
            labels: ['0.1', '0.05', '0.01', '0.005', '0.001']
        },
        ddam: {
            accuracy: [0.9992, 0.9995, 0.9997, 0.99985, 0.999991],
            speed: [450000, 520000, 580000, 640000, 700000],
            error: [1.2e-3, 6e-4, 3e-4, 1.5e-4, 1.156447e-08],
            labels: ['0.1', '0.05', '0.01', '0.005', '0.001']
        }
    };
    
    function updateStats(method) {
        const data = benchmarkData[method];
        if (!data) return;
        
        const avgAccuracy = (data.accuracy.reduce((a, b) => a + b, 0) / data.accuracy.length * 100).toFixed(2);
        const avgSpeed = Math.round(data.speed.reduce((a, b) => a + b, 0) / data.speed.length);
        
        const accuracyEl = document.getElementById('stat-accuracy');
        const speedEl = document.getElementById('stat-speed');
        
        if (accuracyEl) {
            accuracyEl.textContent = avgAccuracy + '%';
        }
        if (speedEl) {
            speedEl.textContent = (avgSpeed >= 1000000) 
                ? (avgSpeed / 1000000).toFixed(1) + 'M'
                : (avgSpeed / 1000).toFixed(0) + 'K';
        }
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
    
    // ============================================================================
    // Lazy Loading for Images
    // ============================================================================
    
    if ('loading' in HTMLImageElement.prototype) {
        const images = document.querySelectorAll('img[loading="lazy"]');
        images.forEach(img => {
            img.src = img.dataset.src || img.src;
        });
    } else {
        // Fallback for browsers that don't support lazy loading
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/lazysizes/5.3.2/lazysizes.min.js';
        document.body.appendChild(script);
    }
    
    // ============================================================================
    // Performance Optimization
    // ============================================================================
    
    // Debounce function for resize events
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // Optimized resize handler
    const handleResize = debounce(() => {
        // Charts will handle their own resize
        if (window.drawComparisonBarCharts) {
            window.drawComparisonBarCharts();
        }
        if (window.drawAccuracySpeedChart) {
            window.drawAccuracySpeedChart();
        }
    }, 250);
    
    window.addEventListener('resize', handleResize, { passive: true });
    
    // ============================================================================
    // Accessibility Enhancements
    // ============================================================================
    
    // Keyboard navigation for buttons
    document.querySelectorAll('.benchmark-btn, .btn').forEach(btn => {
        btn.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });
    });
    
    // Focus management for modals/overlays
    function trapFocus(element) {
        const focusableElements = element.querySelectorAll(
            'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled])'
        );
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];
        
        element.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                if (e.shiftKey && document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                } else if (!e.shiftKey && document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
        });
    }
    
    // ============================================================================
    // Error Handling
    // ============================================================================
    
    window.addEventListener('error', function(e) {
        console.error('JavaScript error:', e.error);
        // Don't break the page on errors
    });
    
    // ============================================================================
    // Export for global access
    // ============================================================================
    
    window.DDRKAM = {
        updateStats: updateStats,
        benchmarkData: benchmarkData
    };
    
    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DDRKAM website initialized');
        });
    } else {
        console.log('DDRKAM website initialized');
    }
})();
