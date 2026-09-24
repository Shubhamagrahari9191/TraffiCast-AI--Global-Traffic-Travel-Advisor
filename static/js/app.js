// =========================================================
// TraffiCast AI - Application State & Logic
// =========================================================

const state = {
    // Current active view: 'global' or 'research'
    activeView: 'global',
    
    // Global Travel Advisor State
    global: {
        origin: { name: 'New York, USA', lat: 40.7128, lng: -74.0060 },
        dest: { name: 'JFK Airport, NY', lat: 40.6413, lng: -73.7781 },
        dayType: 'weekday',
        map: null,
        routeLayers: [],
        originMarker: null,
        destMarker: null,
        chart: null,
        activePinMode: null, // 'origin' | 'dest' | null
    },

    // Research Benchmark State
    research: {
        dataset: 'METR-LA',
        sensorIdx: 0,
        horizonIdx: 0, // 0: 15m, 1: 30m, 2: 60m
        sensors: [],
        chart: null,
        metrics: null,
        significance: null
    }
};

// =========================================================
// Initialization
// =========================================================
document.addEventListener('DOMContentLoaded', () => {
    initViewSwitcher();
    initGlobalAdvisor();
    initResearchDashboard();
});

// =========================================================
// 1. Dual-View Mode Switcher
// =========================================================
function initViewSwitcher() {
    const btnGlobal = document.getElementById('nav-btn-global');
    const btnResearch = document.getElementById('nav-btn-research');
    const viewGlobal = document.getElementById('view-global-advisor');
    const viewResearch = document.getElementById('view-research-hub');
    const sideGlobal = document.getElementById('sidebar-global-controls');
    const sideResearch = document.getElementById('sidebar-research-controls');
    const mainTitle = document.getElementById('page-main-title');
    const mainSubtitle = document.getElementById('page-main-subtitle');

    function switchView(view) {
        state.activeView = view;
        if (view === 'global') {
            btnGlobal.classList.add('active');
            btnResearch.classList.remove('active');
            viewGlobal.classList.add('active');
            viewResearch.classList.remove('active');
            sideGlobal.classList.add('active');
            sideResearch.classList.remove('active');
            
            mainTitle.textContent = "Worldwide Traffic Forecasting & Travel Decision Advisor";
            mainSubtitle.textContent = "Search any location worldwide to analyze 24-hour congestion patterns and find the optimal time to travel.";
            
            // Re-render map tiles to fit dimensions
            if (state.global.map) {
                setTimeout(() => { state.global.map.invalidateSize(); }, 200);
            }
        } else {
            btnResearch.classList.add('active');
            btnGlobal.classList.remove('active');
            viewResearch.classList.add('active');
            viewGlobal.classList.remove('active');
            sideResearch.classList.add('active');
            sideGlobal.classList.remove('active');
            
            mainTitle.textContent = "Spatiotemporal Deep Learning Research Benchmarks";
            mainSubtitle.textContent = "Multi-horizon evaluation of Hybrid Transformer–BiLSTM with Wilcoxon significance & SHAP explainability.";
            
            if (state.research.sensors.length === 0) {
                loadResearchConfig();
            }
        }
    }

    btnGlobal.addEventListener('click', () => switchView('global'));
    btnResearch.addEventListener('click', () => switchView('research'));

    // Benchmark banner jump button
    const btnJumpBenchmark = document.getElementById('btn-jump-benchmark');
    if (btnJumpBenchmark) {
        btnJumpBenchmark.addEventListener('click', () => switchView('research'));
    }
}

// =========================================================
// 2. Global Travel Advisor & Route Engine
// =========================================================
function initGlobalAdvisor() {
    initLeafletMap();
    initAutocomplete();
    initPopularCities();
    initGlobalControls();
    
    // Execute initial route forecast (New York -> JFK)
    calculateGlobalRoute();
}

function initLeafletMap() {
    const mapContainer = document.getElementById('global-map');
    if (!mapContainer) return;

    // Initialize Leaflet map centered at NY
    const map = L.map('global-map', {
        zoomControl: true,
        attributionControl: false
    }).setView([40.7128, -74.0060], 11);

    // CartoDB Dark Matter tile layer for sleek dark theme
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        subdomains: 'abcd',
    }).addTo(map);

    // Map click handler to set Origin / Destination dynamically
    map.on('click', async (e) => {
        const { lat, lng } = e.latlng;
        if (!state.global.origin || (state.global.origin && state.global.dest)) {
            // Set new origin
            state.global.origin = { name: `${lat.toFixed(4)}, ${lng.toFixed(4)}`, lat, lng };
            state.global.dest = null;
            document.getElementById('global-origin-input').value = state.global.origin.name;
            document.getElementById('global-dest-input').value = 'Click map for destination...';
            updateMapMarkers();
        } else if (state.global.origin && !state.global.dest) {
            // Set destination
            state.global.dest = { name: `${lat.toFixed(4)}, ${lng.toFixed(4)}`, lat, lng };
            document.getElementById('global-dest-input').value = state.global.dest.name;
            updateMapMarkers();
            calculateGlobalRoute();
        }
    });

    state.global.map = map;
}

function updateMapMarkers() {
    const map = state.global.map;
    if (!map) return;

    // Remove old markers
    if (state.global.originMarker) map.removeLayer(state.global.originMarker);
    if (state.global.destMarker) map.removeLayer(state.global.destMarker);

    // Custom glowing icons
    const originIcon = L.divIcon({
        className: 'custom-map-pin origin-pin',
        html: `<div style="background: #10b981; width: 14px; height: 14px; border-radius: 50%; border: 3px solid #ffffff; box-shadow: 0 0 14px #10b981;"></div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });

    const destIcon = L.divIcon({
        className: 'custom-map-pin dest-pin',
        html: `<div style="background: #ef4444; width: 14px; height: 14px; border-radius: 50%; border: 3px solid #ffffff; box-shadow: 0 0 14px #ef4444;"></div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });

    if (state.global.origin) {
        state.global.originMarker = L.marker([state.global.origin.lat, state.global.origin.lng], { icon: originIcon })
            .addTo(map)
            .bindPopup(`<b>Origin:</b> ${state.global.origin.name}`);
    }

    if (state.global.dest) {
        state.global.destMarker = L.marker([state.global.dest.lat, state.global.dest.lng], { icon: destIcon })
            .addTo(map)
            .bindPopup(`<b>Destination:</b> ${state.global.dest.name}`);
    }
}

function renderRouteOnMap(routeSegments, coordinates) {
    const map = state.global.map;
    if (!map) return;

    // Clear old route polylines
    state.global.routeLayers.forEach(layer => map.removeLayer(layer));
    state.global.routeLayers = [];

    // Colors according to severity
    const colorMap = {
        free_flow: '#10b981', // Green
        moderate: '#f59e0b',  // Orange
        heavy: '#ef4444'      // Red
    };

    if (routeSegments && routeSegments.length > 0) {
        routeSegments.forEach(seg => {
            const color = colorMap[seg.severity] || '#0ea5e9';
            const polyline = L.polyline(seg.coordinates, {
                color: color,
                weight: 6,
                opacity: 0.9,
                lineCap: 'round',
                lineJoin: 'round'
            }).addTo(map);
            
            polyline.bindPopup(`<b>Segment:</b> ${seg.segment}<br><b>Traffic:</b> ${seg.severity.replace('_', ' ')}`);
            state.global.routeLayers.push(polyline);
        });
    } else if (coordinates && coordinates.length > 0) {
        const polyline = L.polyline(coordinates, {
            color: '#0ea5e9',
            weight: 6,
            opacity: 0.9
        }).addTo(map);
        state.global.routeLayers.push(polyline);
    }

    // Fit map bounds to encompass origin and destination
    if (coordinates && coordinates.length > 0) {
        const bounds = L.latLngBounds(coordinates);
        map.fitBounds(bounds, { padding: [40, 40] });
    }
}

function initGlobalControls() {
    // Route Swap button
    document.getElementById('btn-swap-locations').addEventListener('click', () => {
        const temp = state.global.origin;
        state.global.origin = state.global.dest;
        state.global.dest = temp;

        document.getElementById('global-origin-input').value = state.global.origin ? state.global.origin.name : '';
        document.getElementById('global-dest-input').value = state.global.dest ? state.global.dest.name : '';

        updateMapMarkers();
        calculateGlobalRoute();
    });

    // Day Toggle
    document.querySelectorAll('.btn-day').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-day').forEach(b => b.classList.remove('active'));
            e.currentTarget.classList.add('active');
            state.global.dayType = e.currentTarget.getAttribute('data-day');
            calculateGlobalRoute();
        });
    });

    // Calculate Route Button
    document.getElementById('btn-calculate-route').addEventListener('click', () => {
        calculateGlobalRoute();
    });
}

// Autocomplete geocoding logic
function initAutocomplete() {
    setupAutocomplete('global-origin-input', 'origin-suggestions', (item) => {
        state.global.origin = { name: item.display_name, lat: item.lat, lng: item.lng };
        document.getElementById('global-origin-input').value = item.display_name;
        updateMapMarkers();
    });

    setupAutocomplete('global-dest-input', 'dest-suggestions', (item) => {
        state.global.dest = { name: item.display_name, lat: item.lat, lng: item.lng };
        document.getElementById('global-dest-input').value = item.display_name;
        updateMapMarkers();
    });
}

function setupAutocomplete(inputId, listId, onSelect) {
    const input = document.getElementById(inputId);
    const list = document.getElementById(listId);
    let timeout = null;

    input.addEventListener('input', () => {
        clearTimeout(timeout);
        const q = input.value.trim();
        if (q.length < 2) {
            list.innerHTML = '';
            list.classList.remove('show');
            return;
        }

        timeout = setTimeout(async () => {
            try {
                const res = await fetch(`/api/global/geocode?q=${encodeURIComponent(q)}`);
                const data = await res.json();
                renderSuggestions(data.results || [], list, (item) => {
                    onSelect(item);
                    list.innerHTML = '';
                    list.classList.remove('show');
                });
            } catch (e) {
                console.error("Geocoding failed:", e);
            }
        }, 300);
    });

    // Close suggestions on outside click
    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !list.contains(e.target)) {
            list.classList.remove('show');
        }
    });
}

function renderSuggestions(items, listElem, onSelect) {
    listElem.innerHTML = '';
    if (!items || items.length === 0) {
        listElem.classList.remove('show');
        return;
    }

    items.forEach(item => {
        const li = document.createElement('li');
        li.className = 'autocomplete-item';
        li.innerHTML = `<i class="fa-solid fa-location-dot" style="margin-right: 6px; color: #0ea5e9;"></i> ${item.display_name}`;
        li.addEventListener('click', () => onSelect(item));
        listElem.appendChild(li);
    });
    listElem.classList.add('show');
}

// Popular Cities Quick Selection
async function initPopularCities() {
    const container = document.getElementById('popular-cities-container');
    if (!container) return;

    try {
        const res = await fetch('/api/global/popular-cities');
        const data = await res.json();
        
        container.innerHTML = '';
        data.cities.forEach((city, idx) => {
            const btn = document.createElement('button');
            btn.className = `btn-city-chip ${idx === 0 ? 'active' : ''}`;
            btn.textContent = city.name.split(',')[0];
            btn.title = city.name;
            btn.addEventListener('click', () => {
                document.querySelectorAll('.btn-city-chip').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                state.global.origin = { name: city.name, lat: city.lat, lng: city.lng };
                state.global.dest = city.default_dest;
                
                document.getElementById('global-origin-input').value = city.name;
                document.getElementById('global-dest-input').value = city.default_dest.name;
                
                updateMapMarkers();
                calculateGlobalRoute();
            });
            container.appendChild(btn);
        });
    } catch (e) {
        console.error("Error loading popular cities:", e);
    }
}

// Calculate worldwide route, 24h forecast curve, and travel recommendations
async function calculateGlobalRoute() {
    if (!state.global.origin || !state.global.dest) {
        return;
    }

    const btn = document.getElementById('btn-calculate-route');
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Calculating Forecast...`;
    btn.disabled = true;

    try {
        const url = `/api/global/route?origin_lat=${state.global.origin.lat}&origin_lng=${state.global.origin.lng}`
                  + `&dest_lat=${state.global.dest.lat}&dest_lng=${state.global.dest.lng}`
                  + `&origin_name=${encodeURIComponent(state.global.origin.name)}`
                  + `&dest_name=${encodeURIComponent(state.global.dest.name)}`
                  + `&day_type=${state.global.dayType}`;

        const res = await fetch(url);
        const data = await res.json();

        if (data.success) {
            state.global.currentTrafficProfile = data.traffic_profile;
            updateMapMarkers();
            renderRouteOnMap(data.route_segments, data.coordinates);
            updateGlobalDecisionUI(data);
            renderGlobalTrafficChart(data.traffic_profile);
            initDepartureSlider(data.traffic_profile);
        }
    } catch (e) {
        console.error("Route calculation error:", e);
    } finally {
        btn.innerHTML = `<i class="fa-solid fa-route"></i> Forecast Traffic & Best Time`;
        btn.disabled = false;
    }
}

function updateGlobalDecisionUI(data) {
    const prof = data.traffic_profile;

    // Top Overview Cards
    document.getElementById('global-best-time').textContent = prof.optimal_departure_time;
    document.getElementById('global-time-saved').textContent = `Save up to ${prof.time_saved_min} mins vs peak hours`;

    document.getElementById('global-duration-range').textContent = `${prof.optimal_duration_min} min / ${prof.worst_duration_min} min`;
    document.getElementById('global-distance-text').textContent = `Distance: ${data.distance_km} km (${data.distance_miles} mi)`;

    document.getElementById('global-worst-time').textContent = prof.worst_departure_time;
    document.getElementById('global-peak-window').textContent = `Avoid ${prof.peak_morning_window} & ${prof.peak_evening_window}`;

    // Current Condition Card
    const nowAdvice = prof.leave_now_advice;
    const currentCongestion = nowAdvice.current_congestion;
    document.getElementById('global-current-status').textContent = currentCongestion.label;
    document.getElementById('global-current-duration').textContent = `Duration if you leave now: ${nowAdvice.current_duration_min} min`;

    // Local Time in Advisor Card Header
    const tzSign = nowAdvice.tz_offset_hours >= 0 ? '+' : '';
    document.getElementById('adv-local-time-label').textContent = `Local Time: ${nowAdvice.local_time} (UTC${tzSign}${nowAdvice.tz_offset_hours}) - ${currentCongestion.label}`;

    // Leave Now vs Later Decision Chips
    document.getElementById('chip-now-val').textContent = `${nowAdvice.current_duration_min} min`;
    document.getElementById('chip-30m-val').textContent = `${nowAdvice.leave_in_30m_duration} min`;
    document.getElementById('chip-1h-val').textContent = `${nowAdvice.leave_in_1h_duration} min`;
    document.getElementById('chip-2h-val').textContent = `${nowAdvice.leave_in_2h_duration} min`;

    formatChipDiff('chip-30m-diff', nowAdvice.leave_in_30m_duration - nowAdvice.current_duration_min);
    formatChipDiff('chip-1h-diff', nowAdvice.leave_in_1h_duration - nowAdvice.current_duration_min);
    formatChipDiff('chip-2h-diff', nowAdvice.leave_in_2h_duration - nowAdvice.current_duration_min);

    // Multi-Window best travel times
    document.getElementById('adv-best-morning').textContent = `${nowAdvice.best_morning_time} (${nowAdvice.best_morning_duration} min)`;
    document.getElementById('adv-best-midday').textContent = `${nowAdvice.best_midday_time} (${nowAdvice.best_midday_duration} min)`;
    document.getElementById('adv-best-evening').textContent = `${nowAdvice.best_evening_time} (${nowAdvice.best_evening_duration} min)`;

    // Peak avoidance list
    document.getElementById('adv-morning-peak').textContent = prof.peak_morning_window;
    document.getElementById('adv-evening-peak').textContent = prof.peak_evening_window;

    // Benchmark Network Detection Banner
    const banner = document.getElementById('benchmark-alert-banner');
    const bannerText = document.getElementById('benchmark-banner-text');
    if (data.benchmark_tag) {
        banner.classList.remove('hidden');
        bannerText.textContent = `This route passes through the ${data.benchmark_tag}. High-precision sensor-level forecasts are available.`;
    } else {
        banner.classList.add('hidden');
    }
}

function initDepartureSlider(trafficProfile) {
    const slider = document.getElementById('time-slot-slider');
    if (!slider) return;

    // Set slider initial value to local time slot
    const localTimeParts = (trafficProfile.leave_now_advice.local_time || '10:00').split(':');
    const h = parseInt(localTimeParts[0]);
    const m = parseInt(localTimeParts[1]);
    const initialSlot = Math.min(47, Math.max(0, Math.round((h + m / 60.0) * 2)));
    slider.value = initialSlot;

    function updateSliderDisplay(slotIdx) {
        const timeStr = trafficProfile.time_slots[slotIdx];
        const dur = trafficProfile.durations_min[slotIdx];
        const level = trafficProfile.congestion_levels[slotIdx];

        document.getElementById('slider-time-label').textContent = timeStr;
        document.getElementById('slider-duration-label').textContent = `${dur} min`;
        
        const statusBadge = document.getElementById('slider-status-label');
        statusBadge.textContent = level.label;
        statusBadge.className = `badge-status ${level.level}`;
    }

    updateSliderDisplay(initialSlot);

    slider.oninput = (e) => {
        const slot = parseInt(e.target.value);
        updateSliderDisplay(slot);
    };
}

function formatChipDiff(elementId, diff) {
    const elem = document.getElementById(elementId);
    if (!elem) return;
    const roundDiff = Math.round(diff * 10) / 10;
    if (roundDiff < 0) {
        elem.textContent = `${roundDiff} min (Faster)`;
        elem.className = 'chip-diff faster';
    } else if (roundDiff > 0) {
        elem.textContent = `+${roundDiff} min (Slower)`;
        elem.className = 'chip-diff slower';
    } else {
        elem.textContent = `Same duration`;
        elem.className = 'chip-diff';
    }
}

function renderGlobalTrafficChart(trafficProfile) {
    const ctx = document.getElementById('global-traffic-chart').getContext('2d');
    
    if (state.global.chart) {
        state.global.chart.destroy();
    }

    const labels = trafficProfile.time_slots;
    const durations = trafficProfile.durations_min;
    const speeds = trafficProfile.speeds_kmh;
    const levels = trafficProfile.congestion_levels;

    // Gradient background for travel duration line
    const gradient = ctx.createLinearGradient(0, 0, 0, 320);
    gradient.addColorStop(0, 'rgba(14, 165, 233, 0.45)');
    gradient.addColorStop(1, 'rgba(14, 165, 233, 0.02)');

    state.global.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Predicted Travel Duration (Minutes)',
                    data: durations,
                    borderColor: '#0ea5e9',
                    backgroundColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.35,
                    pointRadius: 2,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#ffffff',
                    pointHoverBorderColor: '#0ea5e9',
                    yAxisID: 'y'
                },
                {
                    label: 'Average Traffic Speed (km/h)',
                    data: speeds,
                    borderColor: '#10b981',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.35,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#9ca3af',
                        font: { family: 'Outfit', size: 12 }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#e2e8f0',
                    borderColor: 'rgba(14, 165, 233, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        afterBody: function(context) {
                            const idx = context[0].dataIndex;
                            const congestion = levels[idx];
                            return `Traffic Status: ${congestion.label}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: {
                        color: '#9ca3af',
                        font: { family: 'Outfit', size: 11 },
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 12
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: {
                        color: '#0ea5e9',
                        font: { family: 'Outfit', size: 11 },
                        callback: value => `${value} min`
                    },
                    title: {
                        display: true,
                        text: 'Travel Duration (min)',
                        color: '#0ea5e9',
                        font: { family: 'Outfit', size: 11 }
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: {
                        color: '#10b981',
                        font: { family: 'Outfit', size: 11 },
                        callback: value => `${value} km/h`
                    },
                    title: {
                        display: true,
                        text: 'Traffic Speed (km/h)',
                        color: '#10b981',
                        font: { family: 'Outfit', size: 11 }
                    }
                }
            }
        }
    });
}

// =========================================================
// 3. Research Benchmark Hub Logic
// =========================================================
function initResearchDashboard() {
    // Dataset switching
    document.querySelectorAll('.btn-dataset').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const button = e.currentTarget;
            document.querySelectorAll('.btn-dataset').forEach(b => b.classList.remove('active'));
            button.classList.add('active');
            state.research.dataset = button.getAttribute('data-dataset');
            state.research.sensorIdx = 0;
            document.getElementById('sensor-search').value = '';
            loadResearchConfig();
        });
    });

    // Horizon switching
    document.querySelectorAll('.btn-horizon').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-horizon').forEach(b => b.classList.remove('active'));
            e.currentTarget.classList.add('active');
            state.research.horizonIdx = parseInt(e.currentTarget.getAttribute('data-horizon'));
            
            const horizonsText = ["15 Minutes Ahead", "30 Minutes Ahead", "60 Minutes Ahead"];
            document.getElementById('summary-horizon').textContent = horizonsText[state.research.horizonIdx];
            
            if (state.research.metrics && state.research.significance) {
                updateSummaryMetrics(state.research.metrics, state.research.significance);
                populateSignificanceTable(state.research.significance);
            }
            
            fetchResearchPredictions();
        });
    });

    // Sensor select
    document.getElementById('sensor-select').addEventListener('change', (e) => {
        state.research.sensorIdx = parseInt(e.target.value);
        fetchResearchPredictions();
    });

    // Sensor search
    document.getElementById('sensor-search').addEventListener('input', (e) => {
        const searchTerm = e.target.value.trim().toLowerCase();
        filterSensorsList(searchTerm);
    });
}

async function loadResearchConfig() {
    try {
        const res = await fetch(`/api/config?dataset=${state.research.dataset}`);
        if (!res.ok) {
            console.error(`Failed to load config: HTTP ${res.status}`);
            return;
        }
        const config = await res.json();
        if (config.error || !config.sensors || !Array.isArray(config.sensors)) {
            console.error("Config load error:", config.error);
            return;
        }
        
        state.research.sensors = config.sensors;
        populateSensorsList(config.sensors);
        
        await fetchResearchMetrics();
        await fetchResearchPredictions();
        await fetchAblationResults();
        updateShapImages();
    } catch (e) {
        console.error("Error loading research config:", e);
    }
}

function populateSensorsList(sensors) {
    const select = document.getElementById('sensor-select');
    select.innerHTML = '';
    sensors.forEach((sensor, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = `Sensor ${sensor}`;
        if (idx === state.research.sensorIdx) opt.selected = true;
        select.appendChild(opt);
    });
}

function filterSensorsList(term) {
    const select = document.getElementById('sensor-select');
    select.innerHTML = '';
    state.research.sensors.forEach((sensor, idx) => {
        if (`${sensor}`.toLowerCase().includes(term) || `sensor ${sensor}`.toLowerCase().includes(term)) {
            const opt = document.createElement('option');
            opt.value = idx;
            opt.textContent = `Sensor ${sensor}`;
            if (idx === state.research.sensorIdx) opt.selected = true;
            select.appendChild(opt);
        }
    });
}

async function fetchResearchPredictions() {
    try {
        const res = await fetch(`/api/predictions?dataset=${state.research.dataset}&sensor=${state.research.sensorIdx}&horizon=${state.research.horizonIdx}`);
        const data = await res.json();
        if (data.error) return;

        document.getElementById('active-sensor-badge').textContent = `Sensor: ${data.sensor_id}`;
        renderResearchChart(data);
    } catch (e) {
        console.error("Error fetching research predictions:", e);
    }
}

function renderResearchChart(data) {
    const ctx = document.getElementById('traffic-chart').getContext('2d');
    
    if (state.research.chart) {
        state.research.chart.destroy();
    }

    const windowSize = Math.min(288, data.actual.length);
    const timestamps = data.timestamps.slice(0, windowSize);
    const actual = data.actual.slice(0, windowSize);

    const datasets = [
        {
            label: 'Actual Ground Truth',
            data: actual,
            borderColor: '#ffffff',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.2
        },
        {
            label: 'Proposed Hybrid (Transformer-BiLSTM)',
            data: data.predictions.hybrid.slice(0, windowSize),
            borderColor: '#10b981',
            borderWidth: 3,
            pointRadius: 0,
            pointHoverRadius: 5,
            tension: 0.2
        },
        {
            label: 'LSTM',
            data: data.predictions.lstm.slice(0, windowSize),
            borderColor: '#60a5fa',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.2
        },
        {
            label: 'GRU',
            data: data.predictions.gru.slice(0, windowSize),
            borderColor: '#fbbf24',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.2
        },
        {
            label: 'BiLSTM',
            data: data.predictions.bilstm.slice(0, windowSize),
            borderColor: '#a78bfa',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.2
        },
        {
            label: 'Transformer',
            data: data.predictions.transformer.slice(0, windowSize),
            borderColor: '#f472b6',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.2
        }
    ];

    if (data.predictions.arima && data.predictions.arima.some(x => x !== null)) {
        datasets.push({
            label: 'ARIMA (Sampled)',
            data: data.predictions.arima.slice(0, windowSize),
            borderColor: '#f87171',
            backgroundColor: '#f87171',
            borderWidth: 0,
            pointRadius: 4,
            pointHoverRadius: 6,
            showLine: false
        });
    }

    state.research.chart = new Chart(ctx, {
        type: 'line',
        data: { labels: timestamps, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: '#9ca3af', font: { family: 'Outfit', size: 11.5 } } }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#9ca3af', maxTicksLimit: 12 }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#9ca3af' },
                    title: { display: true, text: 'Velocity / Flow (mph)', color: '#9ca3af' }
                }
            }
        }
    });
}

async function fetchResearchMetrics() {
    try {
        const res = await fetch(`/api/metrics?dataset=${state.research.dataset}`);
        const data = await res.json();
        state.research.metrics = data.metrics;
        state.research.significance = data.significance;

        populateMetricsTable(data.metrics);
        populateSignificanceTable(data.significance);
        updateSummaryMetrics(data.metrics, data.significance);
    } catch (e) {
        console.error("Error fetching research metrics:", e);
    }
}

function populateMetricsTable(metrics) {
    const tbody = document.querySelector('#metrics-table tbody');
    tbody.innerHTML = '';
    metrics.forEach(m => {
        const tr = document.createElement('tr');
        const isHybrid = m.Model.includes('TRANSFORMER+BiLSTM') || m.Model.includes('Hybrid');
        if (isHybrid) tr.style.background = 'rgba(16, 185, 129, 0.1)';

        tr.innerHTML = `
            <td><strong>${m.Model}</strong></td>
            <td>${m.Horizon}</td>
            <td>${parseFloat(m.MAE).toFixed(2)}</td>
            <td>${parseFloat(m.RMSE).toFixed(2)}</td>
            <td>${parseFloat(m.MAPE).toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
    });
}

function populateSignificanceTable(significance) {
    const tbody = document.querySelector('#significance-table tbody');
    tbody.innerHTML = '';
    const horizonNames = ['15m', '30m', '60m'];
    const activeH = horizonNames[state.research.horizonIdx];

    const filtered = significance.filter(s => s.Horizon === activeH);
    filtered.forEach(s => {
        const tr = document.createElement('tr');
        const isSig = s.Significant === 'True' || s.Significant === true;
        tr.innerHTML = `
            <td>${s.Hypothesis_Pair || s.Baseline_Model}</td>
            <td>${parseFloat(s.Wilcoxon_Statistic).toFixed(1)}</td>
            <td>${parseFloat(s.p_value).toExponential(2)}</td>
            <td><span class="badge" style="background: ${isSig ? 'rgba(16, 185, 129, 0.2); color: #10b981' : 'rgba(239, 68, 68, 0.2); color: #ef4444'}">${isSig ? 'Significant' : 'Not Sig.'}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

function updateSummaryMetrics(metrics, significance) {
    const horizonNames = ['15m', '30m', '60m'];
    const activeH = horizonNames[state.research.horizonIdx];
    document.getElementById('wilcoxon-horizon-label').textContent = activeH;
    document.getElementById('sig-table-horizon-label').textContent = activeH;

    // Calculate MAE improvement vs LSTM for this horizon
    const horizonLabels = ['15 min', '30 min', '60 min'];
    const targetH = horizonLabels[state.research.horizonIdx];
    const hybridRow = metrics.find(m => (m.Model.includes('TRANSFORMER+BiLSTM') || m.Model.includes('Hybrid')) && m.Horizon === targetH);
    const lstmRow = metrics.find(m => m.Model === 'LSTM' && m.Horizon === targetH);

    if (hybridRow && lstmRow) {
        const hybridMAE = parseFloat(hybridRow.MAE);
        const lstmMAE = parseFloat(lstmRow.MAE);
        const imp = ((lstmMAE - hybridMAE) / lstmMAE) * 100;
        document.getElementById('summary-improvement').textContent = `${imp >= 0 ? '-' : '+'}${Math.abs(imp).toFixed(1)}% vs. LSTM`;
    }

    // Wilcoxon p-value for LSTM at this horizon
    const sigRow = significance.find(s => s.Baseline_Model === 'LSTM' && s.Horizon === activeH);
    if (sigRow) {
        const pVal = parseFloat(sigRow.p_value);
        document.getElementById('summary-pvalue').textContent = pVal < 0.001 ? "p < 0.001 (Significant)" : `p = ${pVal.toFixed(3)}`;
    }
}

async function fetchAblationResults() {
    try {
        const res = await fetch(`/api/ablation?dataset=${state.research.dataset}`);
        const data = await res.json();
        const tbody = document.querySelector('#ablation-table tbody');
        tbody.innerHTML = '';
        (data.ablation || []).forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${row.Experiment}</strong></td>
                <td>${parseFloat(row.Overall_MAE).toFixed(2)}</td>
                <td>${parseFloat(row.Overall_RMSE).toFixed(2)}</td>
                <td>${parseFloat(row.Overall_MAPE).toFixed(2)}%</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Error fetching ablation results:", e);
    }
}

function updateShapImages() {
    document.getElementById('shap-img-temporal').src = `/outputs/figures/shap_temporal_importance_Sensor 773869_15m.png?t=${Date.now()}`;
    document.getElementById('shap-img-local').src = `/outputs/figures/shap_local_explanation_Sensor 773869_15m.png?t=${Date.now()}`;
    document.getElementById('shap-img-summary').src = `/outputs/figures/shap_summary_plot_15m.png?t=${Date.now()}`;
}
