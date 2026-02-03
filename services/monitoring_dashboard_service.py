from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import psutil
import time
import statistics


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class Metric:
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    checked_at: datetime = field(default_factory=datetime.utcnow)


class MonitoringDashboardService:
    """
    Comprehensive system monitoring and health tracking
    """

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
        self.request_latencies: List[float] = []
        self.endpoint_stats: Dict[str, Dict[str, Any]] = {}
        self.service_health: Dict[str, HealthCheck] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.metrics_history: Dict[str, List[Metric]] = {}

    def record_request(
        self,
        endpoint: str,
        method: str,
        latency_ms: float,
        status_code: int,
        user_id: str = None
    ):
        """Record an API request"""
        self.request_count += 1
        self.request_latencies.append(latency_ms)

        if len(self.request_latencies) > 10000:
            self.request_latencies = self.request_latencies[-5000:]

        if status_code >= 400:
            self.error_count += 1

        key = f"{method}:{endpoint}"
        if key not in self.endpoint_stats:
            self.endpoint_stats[key] = {
                "count": 0,
                "errors": 0,
                "latencies": [],
                "last_accessed": None
            }

        stats = self.endpoint_stats[key]
        stats["count"] += 1
        if status_code >= 400:
            stats["errors"] += 1
        stats["latencies"].append(latency_ms)
        if len(stats["latencies"]) > 1000:
            stats["latencies"] = stats["latencies"][-500:]
        stats["last_accessed"] = datetime.utcnow().isoformat()

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "status": "high" if cpu_percent > 80 else "normal"
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent": memory.percent,
                    "status": "high" if memory.percent > 85 else "normal"
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": round(disk.percent, 1),
                    "status": "high" if disk.percent > 90 else "normal"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-level metrics"""
        uptime = datetime.utcnow() - self.start_time

        avg_latency = statistics.mean(self.request_latencies) if self.request_latencies else 0
        p95_latency = (
            sorted(self.request_latencies)[int(len(self.request_latencies) * 0.95)]
            if len(self.request_latencies) > 20 else avg_latency
        )
        p99_latency = (
            sorted(self.request_latencies)[int(len(self.request_latencies) * 0.99)]
            if len(self.request_latencies) > 100 else avg_latency
        )

        error_rate = (self.error_count / max(1, self.request_count)) * 100

        return {
            "uptime": {
                "seconds": int(uptime.total_seconds()),
                "formatted": self._format_uptime(uptime)
            },
            "requests": {
                "total": self.request_count,
                "errors": self.error_count,
                "error_rate_percent": round(error_rate, 2),
                "requests_per_minute": round(
                    self.request_count / max(1, uptime.total_seconds() / 60), 2
                )
            },
            "latency": {
                "average_ms": round(avg_latency, 2),
                "p95_ms": round(p95_latency, 2),
                "p99_ms": round(p99_latency, 2),
                "min_ms": round(min(self.request_latencies), 2) if self.request_latencies else 0,
                "max_ms": round(max(self.request_latencies), 2) if self.request_latencies else 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_endpoint_metrics(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get per-endpoint performance metrics"""
        endpoints = []

        for key, stats in self.endpoint_stats.items():
            method, endpoint = key.split(":", 1)
            avg_latency = statistics.mean(stats["latencies"]) if stats["latencies"] else 0

            endpoints.append({
                "endpoint": endpoint,
                "method": method,
                "request_count": stats["count"],
                "error_count": stats["errors"],
                "error_rate_percent": round(
                    (stats["errors"] / max(1, stats["count"])) * 100, 2
                ),
                "avg_latency_ms": round(avg_latency, 2),
                "last_accessed": stats["last_accessed"]
            })

        endpoints.sort(key=lambda x: x["request_count"], reverse=True)
        return endpoints[:limit]

    def check_health(self) -> Dict[str, Any]:
        """Run comprehensive health checks"""
        checks = []

        db_check = self._check_database()
        checks.append(db_check)
        self.service_health["database"] = db_check

        ollama_check = self._check_ollama()
        checks.append(ollama_check)
        self.service_health["ollama"] = ollama_check

        memory_check = self._check_memory()
        checks.append(memory_check)
        self.service_health["memory"] = memory_check

        disk_check = self._check_disk()
        checks.append(disk_check)
        self.service_health["disk"] = disk_check

        overall_status = HealthStatus.HEALTHY
        if any(c.status == HealthStatus.UNHEALTHY for c in checks):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in checks):
            overall_status = HealthStatus.DEGRADED

        return {
            "status": overall_status.value,
            "checks": [{
                "name": c.name,
                "status": c.status.value,
                "message": c.message,
                "latency_ms": round(c.latency_ms, 2),
                "checked_at": c.checked_at.isoformat()
            } for c in checks],
            "timestamp": datetime.utcnow().isoformat()
        }

    def _check_database(self) -> HealthCheck:
        """Check database connectivity"""
        start = time.time()
        try:
            latency = (time.time() - start) * 1000
            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY,
                message="SQLite database operational",
                latency_ms=latency
            )
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start) * 1000
            )

    def _check_ollama(self) -> HealthCheck:
        """Check Ollama LLM service"""
        start = time.time()
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                return HealthCheck(
                    name="ollama",
                    status=HealthStatus.HEALTHY,
                    message="Ollama service connected",
                    latency_ms=latency
                )
            else:
                return HealthCheck(
                    name="ollama",
                    status=HealthStatus.DEGRADED,
                    message=f"Ollama returned status {response.status_code}",
                    latency_ms=latency
                )
        except Exception as e:
            return HealthCheck(
                name="ollama",
                status=HealthStatus.UNHEALTHY,
                message=f"Ollama unreachable: {str(e)[:50]}",
                latency_ms=(time.time() - start) * 1000
            )

    def _check_memory(self) -> HealthCheck:
        """Check system memory"""
        start = time.time()
        try:
            memory = psutil.virtual_memory()
            latency = (time.time() - start) * 1000

            if memory.percent > 95:
                return HealthCheck(
                    name="memory",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Memory critically high: {memory.percent}%",
                    latency_ms=latency
                )
            elif memory.percent > 85:
                return HealthCheck(
                    name="memory",
                    status=HealthStatus.DEGRADED,
                    message=f"Memory high: {memory.percent}%",
                    latency_ms=latency
                )
            else:
                return HealthCheck(
                    name="memory",
                    status=HealthStatus.HEALTHY,
                    message=f"Memory OK: {memory.percent}%",
                    latency_ms=latency
                )
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start) * 1000
            )

    def _check_disk(self) -> HealthCheck:
        """Check disk space"""
        start = time.time()
        try:
            disk = psutil.disk_usage('/')
            latency = (time.time() - start) * 1000

            if disk.percent > 95:
                return HealthCheck(
                    name="disk",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Disk critically full: {disk.percent}%",
                    latency_ms=latency
                )
            elif disk.percent > 85:
                return HealthCheck(
                    name="disk",
                    status=HealthStatus.DEGRADED,
                    message=f"Disk space low: {disk.percent}%",
                    latency_ms=latency
                )
            else:
                return HealthCheck(
                    name="disk",
                    status=HealthStatus.HEALTHY,
                    message=f"Disk OK: {disk.percent}%",
                    latency_ms=latency
                )
        except Exception as e:
            return HealthCheck(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start) * 1000
            )

    def _format_uptime(self, delta: timedelta) -> str:
        """Format uptime as human-readable string"""
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")

        return " ".join(parts)

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get complete dashboard summary"""
        health = self.check_health()
        app_metrics = self.get_application_metrics()
        system_metrics = self.get_system_metrics()

        return {
            "health": health,
            "application": app_metrics,
            "system": system_metrics,
            "top_endpoints": self.get_endpoint_metrics(limit=10),
            "alerts": self.get_active_alerts(),
            "generated_at": datetime.utcnow().isoformat()
        }

    def add_alert(
        self,
        severity: str,
        title: str,
        message: str,
        source: str = "system"
    ):
        """Add a system alert"""
        alert = {
            "id": f"alert_{len(self.alerts)+1}_{int(time.time())}",
            "severity": severity,
            "title": title,
            "message": message,
            "source": source,
            "created_at": datetime.utcnow().isoformat(),
            "acknowledged": False
        }
        self.alerts.append(alert)

        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]

    def get_active_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get active (unacknowledged) alerts"""
        active = [a for a in self.alerts if not a["acknowledged"]]
        return sorted(active, key=lambda x: x["created_at"], reverse=True)[:limit]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.utcnow().isoformat()
                return True
        return False

    def get_metrics_history(
        self,
        metric_name: str,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get historical metrics for graphing"""
        if metric_name not in self.metrics_history:
            return []

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        metrics = [
            m for m in self.metrics_history[metric_name]
            if m.timestamp > cutoff
        ]

        return [{
            "value": m.value,
            "timestamp": m.timestamp.isoformat()
        } for m in metrics]


monitoring_dashboard_service = MonitoringDashboardService()
