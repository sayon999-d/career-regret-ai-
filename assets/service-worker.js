// Career Decision AI - Service Worker
// Enables offline functionality and caching for PWA

const CACHE_NAME = 'career-ai-v2';
const STATIC_CACHE = 'career-ai-static-v2';
const DYNAMIC_CACHE = 'career-ai-dynamic-v2';

// Resources to cache immediately on install
const STATIC_ASSETS = [
    '/',
    '/manifest.json',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    'https://d3js.org/d3.v7.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'
];

// API endpoints that should work offline with cached data
const CACHEABLE_API_PATTERNS = [
    '/api/templates',
    '/api/journal',
    '/api/analytics',
    '/api/goals'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('[Service Worker] Installing...');

    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then((cache) => {
                console.log('[Service Worker] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => {
                console.log('[Service Worker] Static assets cached');
                return self.skipWaiting();
            })
            .catch((err) => {
                console.error('[Service Worker] Cache install failed:', err);
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('[Service Worker] Activating...');

    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => name !== STATIC_CACHE && name !== DYNAMIC_CACHE)
                        .map((name) => {
                            console.log('[Service Worker] Deleting old cache:', name);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => {
                console.log('[Service Worker] Activated');
                return self.clients.claim();
            })
    );
});

// Fetch event - serve from cache when offline
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Skip non-GET requests and non-http/https schemes
    if (request.method !== 'GET' || !url.protocol.startsWith('http')) {
        return;
    }

    // Handle API requests
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(handleApiRequest(request));
        return;
    }

    // Handle static assets with cache-first strategy
    event.respondWith(handleStaticRequest(request));
});

// Cache-first strategy for static assets
async function handleStaticRequest(request) {
    const cachedResponse = await caches.match(request);

    if (cachedResponse) {
        // Return cached version and update in background
        updateCacheInBackground(request);
        return cachedResponse;
    }

    try {
        const networkResponse = await fetch(request);

        if (networkResponse.ok) {
            const cache = await caches.open(STATIC_CACHE);
            cache.put(request, networkResponse.clone());
        }

        return networkResponse;
    } catch (error) {
        // Return offline page if available
        return caches.match('/offline.html') || new Response('Offline', {
            status: 503,
            statusText: 'Service Unavailable'
        });
    }
}

// Network-first strategy for API requests
async function handleApiRequest(request) {
    const url = new URL(request.url);
    const isCacheable = CACHEABLE_API_PATTERNS.some(pattern =>
        url.pathname.startsWith(pattern)
    );

    try {
        const networkResponse = await fetch(request);

        if (networkResponse.ok && isCacheable) {
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }

        return networkResponse;
    } catch (error) {
        // Try to return cached API response (return a clone to avoid locked body streams)
        if (isCacheable) {
            const cachedResponse = await caches.match(request);
            if (cachedResponse) {
                try {
                    return cachedResponse.clone();
                } catch (e) {
                    // Fallback: return original cached response if clone fails
                    return cachedResponse;
                }
            }
        }

        // Return error response for API
        return new Response(JSON.stringify({
            error: 'offline',
            message: 'You are currently offline. Please check your connection.'
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

// Update cache in background (stale-while-revalidate)
async function updateCacheInBackground(request) {
    try {
        const networkResponse = await fetch(request);

        if (networkResponse.ok) {
            const cache = await caches.open(STATIC_CACHE);
            await cache.put(request, networkResponse);
        }
    } catch (error) {
        // Silently fail - we already served from cache
    }
}

// Handle push notifications
self.addEventListener('push', (event) => {
    console.log('[Service Worker] Push received');

    let data = { title: 'Career Decision AI', body: 'New notification' };

    if (event.data) {
        try {
            data = event.data.json();
        } catch (e) {
            data.body = event.data.text();
        }
    }

    const options = {
        body: data.body,
        icon: '/api/pwa/icon-192.png',
        badge: '/api/pwa/badge-72.png',
        vibrate: [100, 50, 100],
        data: data.data || {},
        actions: data.actions || [
            { action: 'open', title: 'Open' },
            { action: 'dismiss', title: 'Dismiss' }
        ],
        tag: data.tag || 'default',
        renotify: data.renotify || false
    };

    event.waitUntil(
        self.registration.showNotification(data.title, options)
    );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
    console.log('[Service Worker] Notification clicked:', event.action);

    event.notification.close();

    if (event.action === 'dismiss') {
        return;
    }

    const urlToOpen = event.notification.data?.url || '/';

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then((clientList) => {
                // Check if app is already open
                for (const client of clientList) {
                    if (client.url.includes(self.location.origin) && 'focus' in client) {
                        client.navigate(urlToOpen);
                        return client.focus();
                    }
                }

                // Open new window
                if (clients.openWindow) {
                    return clients.openWindow(urlToOpen);
                }
            })
    );
});

// Handle background sync
self.addEventListener('sync', (event) => {
    console.log('[Service Worker] Background sync:', event.tag);

    if (event.tag === 'sync-decisions') {
        event.waitUntil(syncPendingDecisions());
    } else if (event.tag === 'sync-journal') {
        event.waitUntil(syncPendingJournalEntries());
    }
});

// Sync pending decisions when back online
async function syncPendingDecisions() {
    const cache = await caches.open(DYNAMIC_CACHE);
    const pendingKey = 'pending-decisions';

    // Get pending decisions from IndexedDB or cache
    // This is a placeholder - would need IndexedDB implementation
    console.log('[Service Worker] Syncing pending decisions...');
}

// Sync pending journal entries when back online
async function syncPendingJournalEntries() {
    console.log('[Service Worker] Syncing pending journal entries...');
}

// Periodic background sync (if supported)
self.addEventListener('periodicsync', (event) => {
    if (event.tag === 'check-opportunities') {
        event.waitUntil(checkForNewOpportunities());
    }
});

async function checkForNewOpportunities() {
    try {
        const response = await fetch('/api/opportunities/check');
        const data = await response.json();

        if (data.new_opportunities > 0) {
            self.registration.showNotification('New Opportunities!', {
                body: `${data.new_opportunities} new opportunities match your profile`,
                icon: '/api/pwa/icon-192.png',
                data: { url: '/?tab=opportunities' }
            });
        }
    } catch (error) {
        console.error('[Service Worker] Opportunity check failed:', error);
    }
}

console.log('[Service Worker] Loaded');
