"""
HPC Monitoring System
Self-contained monitoring for HPC environments with OpenTelemetry-based approaches
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import asyncio
import sqlite3
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    metadata: Dict[str, Any]
    source: str


@dataclass
class APIMetric:
    """API performance metric"""
    timestamp: datetime
    api_name: str
    endpoint: str
    response_time: float
    status_code: int
    success: bool
    error_message: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None


@dataclass
class SystemMetric:
    """System performance metric"""
    timestamp: datetime
    metric_type: str  # cpu, memory, disk, network
    value: float
    unit: str
    node: str
    job_id: Optional[str] = None


class HPCMonitor:
    """
    Self-contained HPC monitoring system
    Provides comprehensive monitoring for scientific computing applications
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HPC monitor

        Args:
            config: Configuration dictionary with monitoring settings
        """
        self.config = config
        self.metrics_db_path = config.get('metrics_db_path', 'data/monitoring/metrics.db')
        self.log_file_path = config.get('log_file_path', 'logs/hpc_monitor.log')
        self.metrics_retention_days = config.get('metrics_retention_days', 30)
        self.metrics_buffer_size = config.get('metrics_buffer_size', 1000)

        # Initialize storage
        self._setup_storage()
        self._setup_logging()

        # Metrics storage
        self.metrics_buffer = deque(maxlen=self.metrics_buffer_size)
        self.api_metrics = defaultdict(list)
        self.system_metrics = defaultdict(list)
        self.performance_metrics = defaultdict(list)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_lock = threading.Lock()

        # Performance tracking
        self.start_time = datetime.now()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        logger.info("HPC Monitor initialized successfully")

    def _setup_storage(self):
        """Setup storage directories and database"""
        # Create directories
        os.makedirs(os.path.dirname(self.metrics_db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        # Initialize SQLite database
        self._init_database()

    def _setup_logging(self):
        """Setup structured logging for monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file_path),
                logging.StreamHandler()
            ]
        )

    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()

                # API metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS api_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        api_name TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        response_time REAL NOT NULL,
                        status_code INTEGER NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        request_size INTEGER,
                        response_size INTEGER,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # System metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        unit TEXT NOT NULL,
                        node TEXT NOT NULL,
                        job_id TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        unit TEXT NOT NULL,
                        metadata TEXT,
                        source TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_metrics_timestamp ON api_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_metrics_api_name ON api_metrics(api_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp)')

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def log_api_performance(self, api_name: str, response_time: float, status_code: int,
                           endpoint: str = "", error_message: str = None,
                           request_size: int = None, response_size: int = None):
        """
        Log API performance metrics for HPC analysis

        Args:
            api_name: Name of the API
            response_time: Response time in seconds
            status_code: HTTP status code
            endpoint: API endpoint
            error_message: Error message if any
            request_size: Size of request in bytes
            response_size: Size of response in bytes
        """
        try:
            metric = APIMetric(
                timestamp=datetime.now(),
                api_name=api_name,
                endpoint=endpoint,
                response_time=response_time,
                status_code=status_code,
                success=200 <= status_code < 300,
                error_message=error_message,
                request_size=request_size,
                response_size=response_size
            )

            # Add to buffer
            with self.metrics_lock:
                self.metrics_buffer.append(metric)
                self.api_metrics[api_name].append(metric)

            # Update counters
            self.total_requests += 1
            if metric.success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            # Log structured metric
            metric_data = {
                'timestamp': metric.timestamp.isoformat(),
                'api': api_name,
                'endpoint': endpoint,
                'response_time': response_time,
                'status': status_code,
                'success': metric.success,
                'error_message': error_message,
                'request_size': request_size,
                'response_size': response_size
            }
            logging.info(json.dumps(metric_data))

            # Persist to database
            self._persist_api_metric(metric)

        except Exception as e:
            logger.error(f"Error logging API performance: {e}")

    def log_system_metric(self, metric_type: str, value: float, unit: str,
                         node: str = "default", job_id: str = None):
        """
        Log system performance metric

        Args:
            metric_type: Type of metric (cpu, memory, disk, network)
            value: Metric value
            unit: Unit of measurement
            node: Node identifier
            job_id: Job ID if applicable
        """
        try:
            metric = SystemMetric(
                timestamp=datetime.now(),
                metric_type=metric_type,
                value=value,
                unit=unit,
                node=node,
                job_id=job_id
            )

            # Add to buffer
            with self.metrics_lock:
                self.metrics_buffer.append(metric)
                self.system_metrics[metric_type].append(metric)

            # Persist to database
            self._persist_system_metric(metric)

        except Exception as e:
            logger.error(f"Error logging system metric: {e}")

    def log_performance_metric(self, metric_name: str, value: float, unit: str,
                              metadata: Dict[str, Any] = None, source: str = "application"):
        """
        Log custom performance metric

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            metadata: Additional metadata
            source: Source of the metric
        """
        try:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_name=metric_name,
                value=value,
                unit=unit,
                metadata=metadata or {},
                source=source
            )

            # Add to buffer
            with self.metrics_lock:
                self.metrics_buffer.append(metric)
                self.performance_metrics[metric_name].append(metric)

            # Persist to database
            self._persist_performance_metric(metric)

        except Exception as e:
            logger.error(f"Error logging performance metric: {e}")

    def _persist_api_metric(self, metric: APIMetric):
        """Persist API metric to database"""
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO api_metrics
                    (timestamp, api_name, endpoint, response_time, status_code, success,
                     error_message, request_size, response_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.api_name,
                    metric.endpoint,
                    metric.response_time,
                    metric.status_code,
                    metric.success,
                    metric.error_message,
                    metric.request_size,
                    metric.response_size
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error persisting API metric: {e}")

    def _persist_system_metric(self, metric: SystemMetric):
        """Persist system metric to database"""
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics
                    (timestamp, metric_type, value, unit, node, job_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.metric_type,
                    metric.value,
                    metric.unit,
                    metric.node,
                    metric.job_id
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error persisting system metric: {e}")

    def _persist_performance_metric(self, metric: PerformanceMetric):
        """Persist performance metric to database"""
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics
                    (timestamp, metric_name, value, unit, metadata, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.metric_name,
                    metric.value,
                    metric.unit,
                    json.dumps(metric.metadata),
                    metric.source
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error persisting performance metric: {e}")

    def get_api_performance_summary(self, api_name: str = None,
                                  time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get API performance summary"""
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()

                # Build query
                query = '''
                    SELECT
                        COUNT(*) as total_requests,
                        AVG(response_time) as avg_response_time,
                        MIN(response_time) as min_response_time,
                        MAX(response_time) as max_response_time,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_requests
                    FROM api_metrics
                    WHERE timestamp >= ?
                '''
                params = [(datetime.now() - time_range).isoformat()]

                if api_name:
                    query += ' AND api_name = ?'
                    params.append(api_name)

                cursor.execute(query, params)
                result = cursor.fetchone()

                if result:
                    total, avg_time, min_time, max_time, success_count, failed_count = result
                    success_rate = (success_count / total * 100) if total > 0 else 0

                    return {
                        'total_requests': total,
                        'successful_requests': success_count,
                        'failed_requests': failed_count,
                        'success_rate': success_rate,
                        'avg_response_time': avg_time,
                        'min_response_time': min_time,
                        'max_response_time': max_time,
                        'time_range': str(time_range)
                    }

                return {}

        except Exception as e:
            logger.error(f"Error getting API performance summary: {e}")
            return {}

    def get_system_metrics_summary(self, metric_type: str = None,
                                 time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get system metrics summary"""
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()

                query = '''
                    SELECT
                        metric_type,
                        AVG(value) as avg_value,
                        MIN(value) as min_value,
                        MAX(value) as max_value,
                        COUNT(*) as sample_count
                    FROM system_metrics
                    WHERE timestamp >= ?
                '''
                params = [(datetime.now() - time_range).isoformat()]

                if metric_type:
                    query += ' AND metric_type = ?'
                    params.append(metric_type)

                query += ' GROUP BY metric_type'

                cursor.execute(query, params)
                results = cursor.fetchall()

                summary = {}
                for row in results:
                    metric_type, avg_val, min_val, max_val, count = row
                    summary[metric_type] = {
                        'avg_value': avg_val,
                        'min_value': min_val,
                        'max_value': max_val,
                        'sample_count': count
                    }

                return summary

        except Exception as e:
            logger.error(f"Error getting system metrics summary: {e}")
            return {}

    def generate_performance_report(self, time_range: timedelta = timedelta(hours=1)) -> str:
        """Generate comprehensive performance report"""
        report = f"""
=== HPC Performance Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Time Range: {time_range}

API PERFORMANCE SUMMARY:
"""

        # API performance
        api_summary = self.get_api_performance_summary(time_range=time_range)
        if api_summary:
            report += f"""
- Total Requests: {api_summary.get('total_requests', 0):,}
- Success Rate: {api_summary.get('success_rate', 0):.2f}%
- Average Response Time: {api_summary.get('avg_response_time', 0):.3f}s
- Min Response Time: {api_summary.get('min_response_time', 0):.3f}s
- Max Response Time: {api_summary.get('max_response_time', 0):.3f}s
"""

        # System metrics
        system_summary = self.get_system_metrics_summary(time_range=time_range)
        if system_summary:
            report += "\nSYSTEM METRICS SUMMARY:\n"
            for metric_type, metrics in system_summary.items():
                report += f"""
{metric_type.upper()}:
- Average: {metrics['avg_value']:.2f}
- Min: {metrics['min_value']:.2f}
- Max: {metrics['max_value']:.2f}
- Samples: {metrics['sample_count']:,}
"""

        # Overall statistics
        uptime = datetime.now() - self.start_time
        report += f"""
OVERALL STATISTICS:
- Uptime: {uptime}
- Total Requests: {self.total_requests:,}
- Successful Requests: {self.successful_requests:,}
- Failed Requests: {self.failed_requests:,}
- Overall Success Rate: {(self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0:.2f}%
"""

        return report

    def export_metrics_to_json(self, output_path: str, time_range: timedelta = timedelta(days=1)):
        """Export metrics to JSON file for external analysis"""
        try:
            metrics_data = {
                'export_timestamp': datetime.now().isoformat(),
                'time_range': str(time_range),
                'api_metrics': [],
                'system_metrics': [],
                'performance_metrics': []
            }

            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()

                # Export API metrics
                cursor.execute('''
                    SELECT * FROM api_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', [(datetime.now() - time_range).isoformat()])

                api_columns = [description[0] for description in cursor.description]
                for row in cursor.fetchall():
                    metrics_data['api_metrics'].append(dict(zip(api_columns, row)))

                # Export system metrics
                cursor.execute('''
                    SELECT * FROM system_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', [(datetime.now() - time_range).isoformat()])

                system_columns = [description[0] for description in cursor.description]
                for row in cursor.fetchall():
                    metrics_data['system_metrics'].append(dict(zip(system_columns, row)))

                # Export performance metrics
                cursor.execute('''
                    SELECT * FROM performance_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', [(datetime.now() - time_range).isoformat()])

                perf_columns = [description[0] for description in cursor.description]
                for row in cursor.fetchall():
                    metrics_data['performance_metrics'].append(dict(zip(perf_columns, row)))

            # Write to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)

            logger.info(f"Metrics exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

    def cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.metrics_retention_days)

            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()

                # Clean up old API metrics
                cursor.execute('DELETE FROM api_metrics WHERE timestamp < ?',
                             [cutoff_date.isoformat()])
                api_deleted = cursor.rowcount

                # Clean up old system metrics
                cursor.execute('DELETE FROM system_metrics WHERE timestamp < ?',
                             [cutoff_date.isoformat()])
                system_deleted = cursor.rowcount

                # Clean up old performance metrics
                cursor.execute('DELETE FROM performance_metrics WHERE timestamp < ?',
                             [cutoff_date.isoformat()])
                perf_deleted = cursor.rowcount

                conn.commit()

                logger.info(f"Cleaned up {api_deleted} API metrics, {system_deleted} system metrics, {perf_deleted} performance metrics")

        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")

    def start_monitoring(self):
        """Start background monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Background monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Background monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Cleanup old metrics periodically
                if datetime.now().hour == 2:  # Run at 2 AM
                    self.cleanup_old_metrics()

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)

    def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.log_system_metric('cpu', cpu_percent, 'percent')

            # Memory usage
            memory = psutil.virtual_memory()
            self.log_system_metric('memory', memory.percent, 'percent')

            # Disk usage
            disk = psutil.disk_usage('/')
            self.log_system_metric('disk', disk.percent, 'percent')

            # Network I/O
            network = psutil.net_io_counters()
            self.log_system_metric('network_bytes_sent', network.bytes_sent, 'bytes')
            self.log_system_metric('network_bytes_recv', network.bytes_recv, 'bytes')

        except ImportError:
            logger.warning("psutil not available, skipping system metrics collection")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'is_monitoring': self.is_monitoring,
            'uptime': str(datetime.now() - self.start_time),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            'metrics_buffer_size': len(self.metrics_buffer),
            'database_path': self.metrics_db_path
        }
