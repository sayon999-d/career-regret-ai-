const CACHE_NAME = 'career-ai-v3';
const STATIC_CACHE = 'career-ai-static-v3';
const DYNAMIC_CACHE = 'career-ai-dynamic-v3';

const STATIC_ASSETS = [
    '/manifest.json',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    'https://d3js.org/d3.v7.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'
];

const CACHEABLE_API_PATTERNS = [
    '/api/templates',
    '/api/journal',
    '/api/analytics',
    '/api/goals'
];

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

self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    if (request.method !== 'GET' || !url.protocol.startsWith('http')) {
        return;
    }

    if (url.pathname.startsWith('/api/')) {
        event.respondWith(handleApiRequest(request));
        return;
    }
    event.respondWith(handleStaticRequest(request));
});

async function handleStaticRequest(request) {
    const acceptsHtml = request.headers.get('accept')?.includes('text/html');
    if (request.mode === 'navigate' || acceptsHtml) {
        return handleNavigationRequest(request);
    }

    const cachedResponse = await caches.match(request);

    if (cachedResponse) {
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
        return caches.match('/offline.html') || new Response('Offline', {
            status: 503,
            statusText: 'Service Unavailable'
        });
    }
}

async function handleNavigationRequest(request) {
    try {
        return await fetch(request, { cache: 'no-store' });
    } catch (error) {
        return caches.match(request) || new Response('Offline', {
            status: 503,
            statusText: 'Service Unavailable'
        });
    }
}

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
        if (isCacheable) {
            const cachedResponse = await caches.match(request);
            if (cachedResponse) {
                try {
                    return cachedResponse.clone();
                } catch (e) {
                    return cachedResponse;
                }
            }
        }
        return new Response(JSON.stringify({
            error: 'offline',
            message: 'You are currently offline. Please check your connection.'
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

async function updateCacheInBackground(request) {
    try {
        const networkResponse = await fetch(request);

        if (networkResponse.ok) {
            const cache = await caches.open(STATIC_CACHE);
            await cache.put(request, networkResponse);
        }
    } catch (error) {
    }
}

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
                for (const client of clientList) {
                    if (client.url.includes(self.location.origin) && 'focus' in client) {
                        client.navigate(urlToOpen);
                        return client.focus();
                    }
                }

                if (clients.openWindow) {
                    return clients.openWindow(urlToOpen);
                }
            })
    );
});

self.addEventListener('sync', (event) => {
    console.log('[Service Worker] Background sync:', event.tag);

    if (event.tag === 'sync-decisions') {
        event.waitUntil(syncPendingDecisions());
    } else if (event.tag === 'sync-journal') {
        event.waitUntil(syncPendingJournalEntries());
    }
});

async function syncPendingDecisions() {
    const cache = await caches.open(DYNAMIC_CACHE);
    const pendingKey = 'pending-decisions';
    console.log('[Service Worker] Syncing pending decisions...');
}

async function syncPendingJournalEntries() {
    console.log('[Service Worker] Syncing pending journal entries...');
}

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
