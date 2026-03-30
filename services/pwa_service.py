from typing import Dict


class PWAService:
    def get_manifest(self) -> Dict:
        return {
            "name": "StepWise AI",
            "short_name": "CareerAI",
            "description": "AI-powered career decision analysis and regret prediction",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#000000",
            "theme_color": "#000000",
            "orientation": "any",
            "icons": [
                {"src": "/assets/icon-192.png", "sizes": "192x192", "type": "image/png"},
                {"src": "/assets/icon-512.png", "sizes": "512x512", "type": "image/png"},
                {"src": "/assets/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}
            ],
            "categories": ["productivity", "education", "business"],
            "screenshots": [],
            "shortcuts": [
                {"name": "New Decision", "url": "/?tab=analysis", "description": "Analyze a new career decision"},
                {"name": "Chat with AI", "url": "/?tab=chat", "description": "Talk to the AI career counselor"},
                {"name": "My Journal", "url": "/?tab=journal", "description": "Review your decision journal"}
            ]
        }

    def get_service_worker_js(self) -> str:
        return '''
const CACHE_NAME = 'stepwise-ai-v2';
const OFFLINE_URL = '/offline.html';

const STATIC_ASSETS = [
    '/',
    '/login',
    '/signup',
    '/offline.html',
    '/api/health'
];

const API_CACHE_PATTERNS = [
    /\\/api\\/journal\\//,
    /\\/api\\/analytics\\//,
    /\\/api\\/gamification\\//,
    /\\/api\\/market\\//
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(STATIC_ASSETS))
            .then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(
                keys.filter(key => key !== CACHE_NAME)
                    .map(key => caches.delete(key))
            )
        ).then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    if (event.request.method !== 'GET') return;

    if (url.pathname.startsWith('/api/')) {
        const shouldCache = API_CACHE_PATTERNS.some(p => p.test(url.pathname));

        event.respondWith(
            fetch(event.request)
                .then(response => {
                    if (shouldCache && response.ok) {
                        const clone = response.clone();
                        caches.open(CACHE_NAME).then(cache => {
                            cache.put(event.request, clone);
                        });
                    }
                    return response;
                })
                .catch(() => caches.match(event.request))
        );
        return;
    }

    event.respondWith(
        caches.match(event.request)
            .then(cached => {
                if (cached) return cached;
                return fetch(event.request)
                    .then(response => {
                        if (response.ok) {
                            const clone = response.clone();
                            caches.open(CACHE_NAME).then(cache => {
                                cache.put(event.request, clone);
                            });
                        }
                        return response;
                    })
                    .catch(() => {
                        if (event.request.headers.get('accept').includes('text/html')) {
                            return caches.match(OFFLINE_URL);
                        }
                    });
            })
    );
});

// Background sync for offline decisions
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-decisions') {
        event.waitUntil(syncOfflineDecisions());
    }
    if (event.tag === 'sync-journal') {
        event.waitUntil(syncOfflineJournal());
    }
});

async function syncOfflineDecisions() {
    try {
        const db = await openDB();
        const tx = db.transaction('offline-decisions', 'readonly');
        const store = tx.objectStore('offline-decisions');
        const decisions = await getAllFromStore(store);

        for (const decision of decisions) {
            try {
                await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(decision.data)
                });
                // Remove synced item
                const deleteTx = db.transaction('offline-decisions', 'readwrite');
                deleteTx.objectStore('offline-decisions').delete(decision.id);
            } catch (e) {
                console.log('Sync failed for decision:', decision.id);
            }
        }
    } catch (e) {
        console.log('Background sync error:', e);
    }
}

async function syncOfflineJournal() {
    try {
        const db = await openDB();
        const tx = db.transaction('offline-journal', 'readonly');
        const store = tx.objectStore('offline-journal');
        const entries = await getAllFromStore(store);

        for (const entry of entries) {
            try {
                await fetch('/api/journal/create', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(entry.data)
                });
                const deleteTx = db.transaction('offline-journal', 'readwrite');
                deleteTx.objectStore('offline-journal').delete(entry.id);
            } catch (e) {
                console.log('Journal sync failed:', entry.id);
            }
        }
    } catch (e) {
        console.log('Journal sync error:', e);
    }
}

function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('CareerAI', 1);
        request.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains('offline-decisions')) {
                db.createObjectStore('offline-decisions', {keyPath: 'id', autoIncrement: true});
            }
            if (!db.objectStoreNames.contains('offline-journal')) {
                db.createObjectStore('offline-journal', {keyPath: 'id', autoIncrement: true});
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

function getAllFromStore(store) {
    return new Promise((resolve, reject) => {
        const request = store.getAll();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

// Push notification handler
self.addEventListener('push', (event) => {
    const data = event.data ? event.data.json() : {};
    const title = data.title || 'StepWise AI';
    const options = {
        body: data.body || 'You have a new notification',
        icon: '/assets/icon-192.png',
        badge: '/assets/icon-192.png',
        data: data.url || '/',
        actions: data.actions || []
    };
    event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    event.waitUntil(
        clients.openWindow(event.notification.data || '/')
    );
});
'''

    def get_offline_html(self) -> str:
        """Return offline fallback page HTML."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StepWise AI - Offline</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            min-height: 100vh; display: flex; align-items: center;
            justify-content: center; background: #f5f5f5; color: #111;
        }
        .offline-container {
            text-align: center; padding: 40px; max-width: 500px;
            background: white; border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .offline-icon { font-size: 48px; margin-bottom: 20px; }
        h1 { font-size: 1.5rem; margin-bottom: 12px; }
        p { color: #666; line-height: 1.6; margin-bottom: 20px; }
        .retry-btn {
            padding: 12px 32px; background: #111; color: white; border: none;
            border-radius: 10px; font-size: 0.9rem; font-weight: 600;
            cursor: pointer; font-family: 'Inter', sans-serif;
        }
        .retry-btn:hover { background: #333; }
        .features { text-align: left; margin: 20px 0; padding: 0 20px; }
        .features li { margin: 8px 0; color: #555; font-size: 0.85rem; }
    </style>
</head>
<body>
    <div class="offline-container">
        <div class="offline-icon">📡</div>
        <h1>You're Offline</h1>
        <p>Don't worry—your data is safe. Some features are available offline:</p>
        <ul class="features">
            <li>View cached journal entries</li>
            <li>Review past decisions and analyses</li>
            <li>Draft new journal entries (will sync later)</li>
            <li>AI chat requires connection</li>
            <li>Live market data unavailable</li>
        </ul>
        <button class="retry-btn" onclick="window.location.reload()">Try Again</button>
    </div>
</body>
</html>'''

    def get_pwa_registration_script(self) -> str:
        return '''
if ('serviceWorker' in navigator) {
    window.addEventListener('load', async () => {
        try {
            const registration = await navigator.serviceWorker.register('/sw.js');
            console.log('SW registered:', registration.scope);

            // Check for updates periodically
            setInterval(() => { registration.update(); }, 60 * 60 * 1000);
        } catch (error) {
            console.log('SW registration failed:', error);
        }
    });
}

// Install prompt handler
let deferredPrompt;
window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;

    // Show install button if exists
    const installBtn = document.getElementById('pwa-install-btn');
    if (installBtn) {
        installBtn.style.display = 'inline-flex';
        installBtn.addEventListener('click', async () => {
            deferredPrompt.prompt();
            const { outcome } = await deferredPrompt.userChoice;
            console.log('Install outcome:', outcome);
            deferredPrompt = null;
            installBtn.style.display = 'none';
        });
    }
});

// Offline data storage helpers
const OfflineStore = {
    async saveDecision(data) {
        try {
            const db = await this._openDB();
            const tx = db.transaction('offline-decisions', 'readwrite');
            tx.objectStore('offline-decisions').add({ data, timestamp: Date.now() });
            return true;
        } catch(e) { return false; }
    },

    async saveJournalEntry(data) {
        try {
            const db = await this._openDB();
            const tx = db.transaction('offline-journal', 'readwrite');
            tx.objectStore('offline-journal').add({ data, timestamp: Date.now() });
            return true;
        } catch(e) { return false; }
    },

    async requestSync(tag) {
        if ('serviceWorker' in navigator && 'SyncManager' in window) {
            const reg = await navigator.serviceWorker.ready;
            await reg.sync.register(tag);
        }
    },

    _openDB() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open('CareerAI', 1);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('offline-decisions'))
                    db.createObjectStore('offline-decisions', {keyPath: 'id', autoIncrement: true});
                if (!db.objectStoreNames.contains('offline-journal'))
                    db.createObjectStore('offline-journal', {keyPath: 'id', autoIncrement: true});
            };
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    }
};

// Online/Offline status indicator
window.addEventListener('online', () => {
    document.body.classList.remove('is-offline');
    const indicator = document.getElementById('connection-status');
    if (indicator) { indicator.textContent = 'Online'; indicator.style.color = '#22c55e'; }
    // Trigger background sync
    OfflineStore.requestSync('sync-decisions');
    OfflineStore.requestSync('sync-journal');
});
window.addEventListener('offline', () => {
    document.body.classList.add('is-offline');
    const indicator = document.getElementById('connection-status');
    if (indicator) { indicator.textContent = 'Offline'; indicator.style.color = '#ef4444'; }
});
'''


pwa_service = PWAService()
