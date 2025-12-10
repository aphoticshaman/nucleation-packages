"""
RYAN-MONITOR 1.0
================
Local training dashboard with email alerts.

No tensorboard. No W&B. Just what you need.

Features:
- Loss spike detection with email alerts
- GPU health monitoring
- Auto cloud burst on thermal throttle
- Simple HTML dashboard
- SQLite backend (no server needed)

Author: Ryan J Cardwell (Archer Phoenix)
"""

import sqlite3
import json
import time
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
import statistics
import subprocess
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MetricPoint:
    timestamp: float
    step: int
    name: str
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    timestamp: float
    severity: str  # 'info', 'warning', 'critical'
    message: str
    metric_name: str
    metric_value: float
    sent: bool = False


@dataclass
class GPUStatus:
    timestamp: float
    gpu_id: int
    temperature: float
    fan_speed: float
    memory_used: float
    memory_total: float
    utilization: float
    power_draw: float


# =============================================================================
# SPIKE DETECTOR
# =============================================================================

class SpikeDetector:
    """
    Detects anomalous spikes in metrics using statistical methods.
    
    Uses rolling window + z-score for spike detection.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        z_threshold: float = 3.0,
        min_samples: int = 20,
    ):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        
        self.windows: Dict[str, deque] = {}
    
    def check(self, name: str, value: float) -> Optional[float]:
        """
        Check if value is a spike.
        
        Returns z-score if spike detected, None otherwise.
        """
        if name not in self.windows:
            self.windows[name] = deque(maxlen=self.window_size)
        
        window = self.windows[name]
        
        if len(window) >= self.min_samples:
            mean = statistics.mean(window)
            std = statistics.stdev(window) if len(window) > 1 else 1.0
            
            if std > 0:
                z_score = (value - mean) / std
                
                if abs(z_score) > self.z_threshold:
                    window.append(value)
                    return z_score
        
        window.append(value)
        return None


# =============================================================================
# EMAIL ALERTER
# =============================================================================

class EmailAlerter:
    """
    Sends email alerts for training events.
    
    Supports SMTP with TLS.
    """
    
    def __init__(
        self,
        smtp_server: str = 'smtp.gmail.com',
        smtp_port: int = 587,
        username: str = '',
        password: str = '',
        from_addr: str = '',
        to_addrs: List[str] = None,
        rate_limit_seconds: float = 60.0,  # Max 1 email per minute
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []
        self.rate_limit_seconds = rate_limit_seconds
        
        self.last_sent: Dict[str, float] = {}
        self.enabled = bool(username and password)
    
    def send(self, subject: str, body: str, category: str = 'default') -> bool:
        """Send an email alert."""
        if not self.enabled:
            print(f"[Alert] {subject}: {body}")
            return False
        
        # Rate limiting
        now = time.time()
        if category in self.last_sent:
            if now - self.last_sent[category] < self.rate_limit_seconds:
                return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)
            msg['Subject'] = f"[RyanMonitor] {subject}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.last_sent[category] = now
            return True
            
        except Exception as e:
            print(f"[EmailAlerter] Failed to send: {e}")
            return False


# =============================================================================
# GPU WATCHDOG
# =============================================================================

class GPUWatchdog:
    """
    Monitors GPU health and triggers actions on thermal/memory issues.
    
    Can trigger cloud burst on thermal throttle.
    """
    
    def __init__(
        self,
        temp_threshold: float = 85.0,
        fan_threshold: float = 90.0,
        memory_threshold: float = 0.95,
        check_interval: float = 5.0,
        on_thermal_alert: Optional[Callable[[], None]] = None,
        on_memory_alert: Optional[Callable[[], None]] = None,
    ):
        self.temp_threshold = temp_threshold
        self.fan_threshold = fan_threshold
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.on_thermal_alert = on_thermal_alert
        self.on_memory_alert = on_memory_alert
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.history: List[GPUStatus] = []
    
    def _get_gpu_status(self) -> List[GPUStatus]:
        """Query GPU status via nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,temperature.gpu,fan.speed,memory.used,memory.total,utilization.gpu,power.draw', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
            )
            
            statuses = []
            now = time.time()
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    statuses.append(GPUStatus(
                        timestamp=now,
                        gpu_id=int(parts[0]),
                        temperature=float(parts[1]),
                        fan_speed=float(parts[2]) if parts[2] != '[N/A]' else 0,
                        memory_used=float(parts[3]),
                        memory_total=float(parts[4]),
                        utilization=float(parts[5]),
                        power_draw=float(parts[6]) if parts[6] != '[N/A]' else 0,
                    ))
            
            return statuses
            
        except Exception as e:
            print(f"[GPUWatchdog] Failed to query GPU: {e}")
            return []
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            statuses = self._get_gpu_status()
            self.history.extend(statuses)
            
            # Keep history bounded
            if len(self.history) > 10000:
                self.history = self.history[-5000:]
            
            for status in statuses:
                # Check temperature
                if status.temperature > self.temp_threshold:
                    if self.on_thermal_alert:
                        self.on_thermal_alert()
                
                # Check fan speed (indicates thermal stress)
                if status.fan_speed > self.fan_threshold:
                    if self.on_thermal_alert:
                        self.on_thermal_alert()
                
                # Check memory
                mem_usage = status.memory_used / status.memory_total
                if mem_usage > self.memory_threshold:
                    if self.on_memory_alert:
                        self.on_memory_alert()
            
            time.sleep(self.check_interval)
    
    def start(self):
        """Start monitoring."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def get_current_status(self) -> List[GPUStatus]:
        """Get current GPU status."""
        return self._get_gpu_status()


# =============================================================================
# CLOUD BURST MANAGER
# =============================================================================

class CloudBurstManager:
    """
    Manages cloud burst for failover.
    
    When GPU hits thermal limit:
    1. Save state
    2. Spin up cloud instance
    3. Transfer state
    4. Resume training
    """
    
    def __init__(
        self,
        provider: str = 'aws',  # 'aws', 'gcp', 'azure', 'lambda'
        instance_type: str = 'p4d.24xlarge',
        region: str = 'us-west-2',
        state_bucket: str = '',
        ssh_key_path: str = '',
    ):
        self.provider = provider
        self.instance_type = instance_type
        self.region = region
        self.state_bucket = state_bucket
        self.ssh_key_path = ssh_key_path
        
        self.instance_id: Optional[str] = None
        self.instance_ip: Optional[str] = None
    
    def save_state(self, state_dict: Dict, path: str) -> str:
        """Save training state to cloud storage."""
        import torch
        
        # Save locally first
        torch.save(state_dict, path)
        
        # Upload to cloud
        if self.provider == 'aws' and self.state_bucket:
            try:
                import boto3
                s3 = boto3.client('s3')
                key = f"training_states/{Path(path).name}"
                s3.upload_file(path, self.state_bucket, key)
                return f"s3://{self.state_bucket}/{key}"
            except Exception as e:
                print(f"[CloudBurst] Failed to upload state: {e}")
        
        return path
    
    def spin_up_instance(self) -> bool:
        """Spin up a cloud GPU instance."""
        if self.provider == 'aws':
            try:
                import boto3
                ec2 = boto3.client('ec2', region_name=self.region)
                
                response = ec2.run_instances(
                    ImageId='ami-0123456789abcdef0',  # Deep Learning AMI
                    InstanceType=self.instance_type,
                    MinCount=1,
                    MaxCount=1,
                    KeyName=Path(self.ssh_key_path).stem if self.ssh_key_path else None,
                )
                
                self.instance_id = response['Instances'][0]['InstanceId']
                
                # Wait for running
                waiter = ec2.get_waiter('instance_running')
                waiter.wait(InstanceIds=[self.instance_id])
                
                # Get IP
                response = ec2.describe_instances(InstanceIds=[self.instance_id])
                self.instance_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
                
                print(f"[CloudBurst] Instance {self.instance_id} running at {self.instance_ip}")
                return True
                
            except Exception as e:
                print(f"[CloudBurst] Failed to spin up: {e}")
                return False
        
        print(f"[CloudBurst] Provider {self.provider} not implemented")
        return False
    
    def transfer_and_resume(self, state_path: str, script_path: str) -> bool:
        """Transfer state to cloud instance and resume training."""
        if not self.instance_ip:
            return False
        
        try:
            # SCP state file
            subprocess.run([
                'scp', '-i', self.ssh_key_path,
                state_path,
                f'ubuntu@{self.instance_ip}:/home/ubuntu/state.pt'
            ], check=True)
            
            # SCP training script
            subprocess.run([
                'scp', '-i', self.ssh_key_path,
                script_path,
                f'ubuntu@{self.instance_ip}:/home/ubuntu/train.py'
            ], check=True)
            
            # Start training
            subprocess.run([
                'ssh', '-i', self.ssh_key_path,
                f'ubuntu@{self.instance_ip}',
                'nohup python /home/ubuntu/train.py --resume /home/ubuntu/state.pt &'
            ], check=True)
            
            print(f"[CloudBurst] Training resumed on {self.instance_ip}")
            return True
            
        except Exception as e:
            print(f"[CloudBurst] Transfer failed: {e}")
            return False
    
    def terminate_instance(self):
        """Terminate the cloud instance."""
        if self.instance_id and self.provider == 'aws':
            try:
                import boto3
                ec2 = boto3.client('ec2', region_name=self.region)
                ec2.terminate_instances(InstanceIds=[self.instance_id])
                print(f"[CloudBurst] Terminated {self.instance_id}")
            except Exception as e:
                print(f"[CloudBurst] Failed to terminate: {e}")


# =============================================================================
# SQLITE BACKEND
# =============================================================================

class MetricsDB:
    """SQLite backend for metrics storage."""
    
    def __init__(self, path: str = 'metrics.db'):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_tables()
    
    def _init_tables(self):
        with self.lock:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    step INTEGER,
                    name TEXT,
                    value REAL,
                    tags TEXT
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    severity TEXT,
                    message TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    sent INTEGER
                )
            ''')
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_name_step 
                ON metrics(name, step)
            ''')
            self.conn.commit()
    
    def log_metric(self, point: MetricPoint):
        with self.lock:
            self.conn.execute(
                'INSERT INTO metrics (timestamp, step, name, value, tags) VALUES (?, ?, ?, ?, ?)',
                (point.timestamp, point.step, point.name, point.value, json.dumps(point.tags))
            )
            self.conn.commit()
    
    def log_alert(self, alert: Alert):
        with self.lock:
            self.conn.execute(
                'INSERT INTO alerts (timestamp, severity, message, metric_name, metric_value, sent) VALUES (?, ?, ?, ?, ?, ?)',
                (alert.timestamp, alert.severity, alert.message, alert.metric_name, alert.metric_value, int(alert.sent))
            )
            self.conn.commit()
    
    def get_metrics(self, name: str, start_step: int = 0, end_step: int = None) -> List[MetricPoint]:
        with self.lock:
            if end_step is None:
                cursor = self.conn.execute(
                    'SELECT timestamp, step, name, value, tags FROM metrics WHERE name = ? AND step >= ? ORDER BY step',
                    (name, start_step)
                )
            else:
                cursor = self.conn.execute(
                    'SELECT timestamp, step, name, value, tags FROM metrics WHERE name = ? AND step >= ? AND step <= ? ORDER BY step',
                    (name, start_step, end_step)
                )
            
            return [
                MetricPoint(
                    timestamp=row[0],
                    step=row[1],
                    name=row[2],
                    value=row[3],
                    tags=json.loads(row[4]) if row[4] else {}
                )
                for row in cursor.fetchall()
            ]
    
    def get_latest(self, name: str) -> Optional[MetricPoint]:
        with self.lock:
            cursor = self.conn.execute(
                'SELECT timestamp, step, name, value, tags FROM metrics WHERE name = ? ORDER BY step DESC LIMIT 1',
                (name,)
            )
            row = cursor.fetchone()
            if row:
                return MetricPoint(
                    timestamp=row[0],
                    step=row[1],
                    name=row[2],
                    value=row[3],
                    tags=json.loads(row[4]) if row[4] else {}
                )
            return None


# =============================================================================
# MAIN MONITOR
# =============================================================================

class RyanMonitor:
    """
    Main training monitor.
    
    Usage:
        monitor = RyanMonitor(
            email_config={'smtp_server': '...', 'username': '...', 'password': '...'},
        )
        monitor.start()
        
        # In training loop:
        monitor.log('loss', loss_value, step=step)
        monitor.log('lr', lr_value, step=step)
        
        # Monitor handles:
        # - Spike detection
        # - Email alerts
        # - GPU health
        # - Dashboard
    """
    
    def __init__(
        self,
        db_path: str = 'metrics.db',
        email_config: Dict = None,
        gpu_config: Dict = None,
        dashboard_port: int = 8080,
        spike_threshold: float = 3.0,
    ):
        # Components
        self.db = MetricsDB(db_path)
        self.spike_detector = SpikeDetector(z_threshold=spike_threshold)
        
        # Email
        email_config = email_config or {}
        self.emailer = EmailAlerter(**email_config)
        
        # GPU watchdog
        gpu_config = gpu_config or {}
        self.gpu_watchdog = GPUWatchdog(
            on_thermal_alert=self._on_thermal,
            on_memory_alert=self._on_memory,
            **gpu_config
        )
        
        # Dashboard
        self.dashboard_port = dashboard_port
        self.dashboard_thread: Optional[threading.Thread] = None
        
        # State
        self.current_step = 0
        self.start_time = time.time()
    
    def start(self):
        """Start all monitoring components."""
        self.gpu_watchdog.start()
        self._start_dashboard()
        print(f"[RyanMonitor] Started. Dashboard at http://localhost:{self.dashboard_port}")
    
    def stop(self):
        """Stop all monitoring components."""
        self.gpu_watchdog.stop()
    
    def log(self, name: str, value: float, step: int = None, **tags):
        """Log a metric value."""
        if step is not None:
            self.current_step = step
        
        point = MetricPoint(
            timestamp=time.time(),
            step=self.current_step,
            name=name,
            value=value,
            tags=tags,
        )
        
        self.db.log_metric(point)
        
        # Check for spike
        z_score = self.spike_detector.check(name, value)
        if z_score is not None:
            self._handle_spike(name, value, z_score)
    
    def _handle_spike(self, name: str, value: float, z_score: float):
        """Handle a detected spike."""
        severity = 'critical' if abs(z_score) > 5 else 'warning'
        message = f"Spike detected in {name}: value={value:.4f}, z-score={z_score:.2f}"
        
        alert = Alert(
            timestamp=time.time(),
            severity=severity,
            message=message,
            metric_name=name,
            metric_value=value,
        )
        
        self.db.log_alert(alert)
        
        # Send email
        if severity == 'critical' or (severity == 'warning' and name == 'loss'):
            sent = self.emailer.send(
                subject=f"Loss Spike Detected (step {self.current_step})",
                body=message,
                category=f"spike_{name}"
            )
            alert.sent = sent
    
    def _on_thermal(self):
        """Handle thermal alert."""
        self.emailer.send(
            subject="GPU Thermal Alert",
            body=f"GPU temperature exceeds threshold at step {self.current_step}",
            category="thermal"
        )
    
    def _on_memory(self):
        """Handle memory alert."""
        self.emailer.send(
            subject="GPU Memory Alert",
            body=f"GPU memory usage exceeds threshold at step {self.current_step}",
            category="memory"
        )
    
    def _start_dashboard(self):
        """Start the dashboard server."""
        # Generate dashboard HTML
        self._generate_dashboard_html()
        
        # Start server in background
        def serve():
            os.chdir(Path(self.db.path).parent)
            handler = SimpleHTTPRequestHandler
            with HTTPServer(('', self.dashboard_port), handler) as server:
                server.serve_forever()
        
        self.dashboard_thread = threading.Thread(target=serve, daemon=True)
        self.dashboard_thread.start()
    
    def _generate_dashboard_html(self):
        """Generate dashboard HTML file."""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>RyanMonitor Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        h1 { color: #00ff88; }
        .chart { width: 100%; height: 400px; margin: 20px 0; }
        .stats { display: flex; gap: 20px; }
        .stat-box { background: #2a2a2a; padding: 20px; border-radius: 8px; }
        .stat-value { font-size: 32px; color: #00ff88; }
        .stat-label { color: #888; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .alert-warning { background: #554400; }
        .alert-critical { background: #550000; }
    </style>
</head>
<body>
    <h1>🔥 RyanMonitor</h1>
    <div class="stats">
        <div class="stat-box">
            <div class="stat-value" id="current-step">0</div>
            <div class="stat-label">Current Step</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="current-loss">-</div>
            <div class="stat-label">Loss</div>
        </div>
        <div class="stat-box">
            <div class="stat-value" id="gpu-temp">-</div>
            <div class="stat-label">GPU Temp</div>
        </div>
    </div>
    <div id="loss-chart" class="chart"></div>
    <div id="lr-chart" class="chart"></div>
    <h2>Alerts</h2>
    <div id="alerts"></div>
    <script>
        // Auto-refresh every 5 seconds
        setInterval(updateDashboard, 5000);
        function updateDashboard() {
            // Fetch metrics from API endpoint (would need backend)
            console.log('Dashboard refresh');
        }
    </script>
</body>
</html>'''
        
        with open('dashboard.html', 'w') as f:
            f.write(html)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RyanMonitor',
    'SpikeDetector',
    'EmailAlerter',
    'GPUWatchdog',
    'CloudBurstManager',
    'MetricsDB',
    'MetricPoint',
    'Alert',
    'GPUStatus',
]


if __name__ == "__main__":
    # Quick test
    monitor = RyanMonitor(db_path='/tmp/test_metrics.db')
    monitor.start()
    
    import random
    for step in range(100):
        loss = 2.0 / (step + 1) + random.gauss(0, 0.1)
        monitor.log('loss', loss, step=step)
        
        # Inject spike
        if step == 50:
            monitor.log('loss', 10.0, step=step)
        
        time.sleep(0.01)
    
    print("Test complete. Check /tmp/test_metrics.db")
