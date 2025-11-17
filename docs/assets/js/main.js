// Main JavaScript for DDRKAM website
// Copyright (C) 2025, Shyamal Suhana Chandra
//
// Implementation Credits:
// - Intersection Observer API: W3C Standard [9]
// - Material Design Ripple Effect: Google Material Design [8]
// - Glassmorphism Effects: Apple Human Interface Guidelines [7]
// - Modern Web Animations: MDN Web Docs [10]

(function() {
    'use strict';
    
    // ============================================================================
    // Loading & Initialization
    // ============================================================================

    // Loading screen
    function showLoadingScreen() {
        const loadingScreen = document.createElement('div');
        loadingScreen.id = 'loading-screen';
        loadingScreen.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--bg-darker);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            transition: opacity 0.5s ease-out;
        `;

        loadingScreen.innerHTML = `
            <div style="text-align: center;">
                <div style="
                    width: 60px;
                    height: 60px;
                    border: 3px solid rgba(99, 102, 241, 0.3);
                    border-radius: 50%;
                    border-top-color: var(--primary-color);
                    animation: rotate 1s linear infinite;
                    margin: 0 auto 1rem;
                "></div>
                <h2 style="
                    background: var(--gradient-1);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    font-size: 1.5rem;
                    margin: 0;
                ">DDRKAM</h2>
                <p style="color: var(--text-secondary); margin: 0.5rem 0 0;">Loading...</p>
            </div>
        `;

        document.body.appendChild(loadingScreen);

        // Hide loading screen after content loads
        window.addEventListener('load', () => {
            setTimeout(() => {
                loadingScreen.style.opacity = '0';
                setTimeout(() => {
                    loadingScreen.remove();
                    initializeAnimations();
                }, 500);
            }, 1000);
        });
    }

    // Initialize scroll-triggered animations
    function initializeAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const sectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    // Stagger child animations
                    const children = entry.target.querySelectorAll('.feature-card, .doc-card, .stat-card, .chart-container');
                    children.forEach((child, index) => {
                        setTimeout(() => {
                            child.classList.add('fade-in-up');
                        }, index * 100);
                    });
                }
            });
        }, observerOptions);

        // Observe all sections
        document.querySelectorAll('section').forEach(section => {
            sectionObserver.observe(section);
        });

        // Add smooth reveal for hero elements
        const heroElements = document.querySelectorAll('.hero-title, .hero-subtitle, .hero-buttons');
        heroElements.forEach((element, index) => {
            setTimeout(() => {
                element.style.animation = 'fadeInUp 0.8s ease-out forwards';
            }, index * 200);
        });

        // Initialize enhanced interactions
        enhanceButtonInteractions();
        enhanceCardInteractions();
        enhanceChartLoading();
        initializeThemeToggle();
    }

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
    
    // ============================================================================
    // Intersection Observer for active nav links
    // ============================================================================

    const sections = document.querySelectorAll('section[id]');
    const navObserverOptions = {
        root: null,
        rootMargin: '-20% 0px -70% 0px',
        threshold: 0
    };

    const navObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                updateActiveNavLink('#' + id);
            }
        });
    }, navObserverOptions);

    sections.forEach(section => navObserver.observe(section));
    
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
    // Theme Toggle Functionality
    // ============================================================================

    function initializeThemeToggle() {
        // Create theme toggle button
        const themeToggle = document.createElement('button');
        themeToggle.className = 'theme-toggle';
        themeToggle.setAttribute('aria-label', 'Toggle theme');
        themeToggle.innerHTML = `
            <svg class="theme-toggle-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 3V4M12 20V21M4 12H3M6.31412 6.31412L5.5 5.5M17.6859 6.31412L18.5 5.5M6.31412 17.69L5.5 18.5M17.6859 17.69L18.5 18.5M21 12H20M16 12C16 14.2091 14.2091 16 12 16C9.79086 16 8 14.2091 8 12C8 9.79086 9.79086 8 12 8C14.2091 8 16 9.79086 16 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M12 3V4M12 20V21M4 12H3M21 12H20M16 12C16 14.2091 14.2091 16 12 16C9.79086 16 8 14.2091 8 12C8 9.79086 9.79086 8 12 8C14.2091 8 16 9.79086 16 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="sun-rays"/>
                <circle cx="12" cy="12" r="4" fill="currentColor" class="sun-circle"/>
                <path d="M12 3C7.02944 3 3 7.02944 3 12C3 16.9706 7.02944 21 12 21C14.4853 21 16.7353 20.0571 18.364 18.636L16.9497 17.2217C15.6141 18.251 13.8774 18.9 12 18.9C8.0257 18.9 4.8 15.6743 4.8 11.7C4.8 7.7257 8.0257 4.5 12 4.5C13.8774 4.5 15.6141 5.14903 16.9497 6.1783L18.364 4.764C16.7353 3.34289 14.4853 2.4 12 2.4V3Z" fill="currentColor" class="moon-path"/>
            </svg>
        `;

        document.body.appendChild(themeToggle);

        // Check for saved theme preference or default to dark mode
        const savedTheme = localStorage.getItem('ddrkam-theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.classList.add('light-mode');
            updateThemeIcon(true);
        }

        // Toggle theme on click
        themeToggle.addEventListener('click', () => {
            const isLight = document.body.classList.toggle('light-mode');
            localStorage.setItem('ddrkam-theme', isLight ? 'light' : 'dark');
            updateThemeIcon(isLight);

            // Update charts if they exist
            if (window.updateCharts) {
                const activeBtn = document.querySelector('.benchmark-btn.active');
                if (activeBtn) {
                    window.updateCharts(activeBtn.dataset.method);
                }
            }
        });

        function updateThemeIcon(isLight) {
            const icon = themeToggle.querySelector('.theme-toggle-icon');
            if (isLight) {
                // Show moon icon for light mode (switch to dark)
                icon.innerHTML = `
                    <path d="M12 3C7.02944 3 3 7.02944 3 12C3 16.9706 7.02944 21 12 21C14.4853 21 16.7353 20.0571 18.364 18.636L16.9497 17.2217C15.6141 18.251 13.8774 18.9 12 18.9C8.0257 18.9 4.8 15.6743 4.8 11.7C4.8 7.7257 8.0257 4.5 12 4.5C13.8774 4.5 15.6141 5.14903 16.9497 6.1783L18.364 4.764C16.7353 3.34289 14.4853 2.4 12 2.4V3Z" fill="currentColor"/>
                `;
            } else {
                // Show sun icon for dark mode (switch to light)
                icon.innerHTML = `
                    <circle cx="12" cy="12" r="4" fill="currentColor"/>
                    <path d="M12 3V4M12 20V21M4 12H3M6.31412 6.31412L5.5 5.5M17.6859 6.31412L18.5 5.5M6.31412 17.69L5.5 18.5M17.6859 17.69L18.5 18.5M21 12H20M16 12C16 14.2091 14.2091 16 12 16C9.79086 16 8 14.2091 8 12C8 9.79086 9.79086 8 12 8C14.2091 8 16 9.79086 16 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                `;
            }
        }
    }

    // ============================================================================
    // Enhanced Interactive Elements
    // ============================================================================

    // Button click animations
    function enhanceButtonInteractions() {
        document.querySelectorAll('.btn').forEach(btn => {
            btn.addEventListener('click', function(e) {
                // Create ripple effect
                const ripple = document.createElement('span');
                ripple.style.cssText = `
                    position: absolute;
                    border-radius: 50%;
                    background: rgba(255, 255, 255, 0.3);
                    transform: scale(0);
                    animation: ripple 0.6s linear;
                    pointer-events: none;
                `;

                const rect = this.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;

                ripple.style.width = ripple.style.height = size + 'px';
                ripple.style.left = x + 'px';
                ripple.style.top = y + 'px';

                this.appendChild(ripple);

                setTimeout(() => ripple.remove(), 600);
            });
        });

        // Add CSS for ripple animation
        if (!document.getElementById('ripple-styles')) {
            const style = document.createElement('style');
            style.id = 'ripple-styles';
            style.textContent = `
                @keyframes ripple {
                    to {
                        transform: scale(4);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    // Enhanced hover effects for cards
    function enhanceCardInteractions() {
        document.querySelectorAll('.feature-card, .doc-card, .stat-card').forEach(card => {
            card.addEventListener('mouseenter', function() {
                // Add subtle glow effect
                this.style.boxShadow = `0 20px 40px rgba(0, 0, 0, 0.2), var(--shadow-glow)`;
            });

            card.addEventListener('mouseleave', function() {
                // Reset shadow
                this.style.boxShadow = '';
            });
        });
    }

    // Dynamic content loading for benchmark charts
    function enhanceChartLoading() {
        const chartContainers = document.querySelectorAll('.chart-container');

        chartContainers.forEach(container => {
            // Add loading state
            container.classList.add('loading');

            // Simulate loading delay for demo
            setTimeout(() => {
                container.classList.remove('loading');
                container.classList.add('loaded');
            }, Math.random() * 1000 + 500);
        });
    }

    // ============================================================================
    // Benchmark Statistics Update
    // ============================================================================
    
    // Validated benchmark data from comprehensive test suite
    // Copyright (C) 2025, Shyamal Suhana Chandra
    const benchmarkData = {
        euler: {
            accuracy: [0.9900, 0.9950, 0.9980, 0.9990, 0.9995],
            speed: [1500000, 1600000, 1700000, 1800000, 1900000],
            error: [1e-2, 5e-3, 2e-3, 1e-3, 5e-4],
            labels: ['0.1', '0.05', '0.01', '0.005', '0.001']
        },
        ddeuler: {
            accuracy: [0.9920, 0.9960, 0.9985, 0.9992, 0.9996],
            speed: [800000, 850000, 900000, 950000, 1000000],
            error: [8e-3, 4e-3, 1.5e-3, 8e-4, 4e-4],
            labels: ['0.1', '0.05', '0.01', '0.005', '0.001']
        },
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
    
    // ============================================================================
    // Initialize Application
    // ============================================================================

    // Start loading screen
    showLoadingScreen();

    // ============================================================================
    // Accordion Functionality for Complexity Proofs
    // ============================================================================
    
    function initAccordions() {
        const accordionHeaders = document.querySelectorAll('.accordion-header');
        
        accordionHeaders.forEach(header => {
            header.addEventListener('click', function() {
                const isExpanded = this.getAttribute('aria-expanded') === 'true';
                const content = this.nextElementSibling;
                
                // Toggle current accordion
                this.setAttribute('aria-expanded', !isExpanded);
                
                if (!isExpanded) {
                    // Expand accordion
                    content.style.maxHeight = content.scrollHeight + 'px';
                    
                    // Wait for transition, then render MathJax
                    setTimeout(function() {
                        if (window.MathJax) {
                            // Check if MathJax is loaded and ready
                            if (window.MathJax.typesetPromise) {
                                window.MathJax.typesetPromise([content]).then(function() {
                                    // Adjust height after MathJax renders (formulas may change height)
                                    content.style.maxHeight = content.scrollHeight + 'px';
                                }).catch(function (err) {
                                    console.error('MathJax rendering error:', err);
                                });
                            } else if (window.MathJax.startup && window.MathJax.startup.promise) {
                                // MathJax not ready yet, wait for it
                                window.MathJax.startup.promise.then(function() {
                                    if (window.MathJax.typesetPromise) {
                                        window.MathJax.typesetPromise([content]).then(function() {
                                            content.style.maxHeight = content.scrollHeight + 'px';
                                        }).catch(function (err) {
                                            console.error('MathJax rendering error:', err);
                                        });
                                    }
                                });
                            }
                        }
                    }, 150);
                } else {
                    // Collapse accordion
                    content.style.maxHeight = '0';
                }
            });
        });
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DDRKAM website DOM loaded');
            initAccordions();
        });
    } else {
        console.log('DDRKAM website initialized');
        initAccordions();
    }
})();
