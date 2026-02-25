"""
Automated Scheduler for Production Analysis
Runs analysis at scheduled intervals or specific times
"""

import schedule
import time
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnalysisScheduler:
    """Schedule and automate analysis runs"""
    
    def __init__(self, config_file: str = 'scheduler_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
        self.last_run = None
        self.run_history = []
        
    def load_config(self) -> Dict:
        """Load scheduler configuration"""
        default_config = {
            'schedule': {
                'daily_time': '02:00',  # Run at 2 AM daily
                'weekly_day': 'sunday',  # Run weekly on Sunday
                'monthly_day': 1,  # Run on 1st of each month
                'mode': 'daily'  # daily, weekly, monthly, or manual
            },
            'notifications': {
                'enabled': False,
                'email': '',
                'smtp_server': '',
                'smtp_port': 587,
                'smtp_user': '',
                'smtp_password': ''
            },
            'retention': {
                'keep_days': 30,  # Keep results for 30 days
                'max_runs': 10  # Keep maximum 10 runs
            },
            'analysis_options': {
                'batch_size': 1000,
                'max_workers': 10,
                'generate_visualizations': True,
                'save_to_gcs': True
            }
        }
        
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Save default config
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def run_analysis(self) -> bool:
        """Execute the production analysis"""
        logger.info("="*60)
        logger.info("Starting scheduled analysis run")
        logger.info("="*60)
        
        start_time = datetime.now()
        success = False
        
        try:
            # Run the production deployment script
            result = subprocess.run(
                ['python', 'deploy_full_analysis.py'],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            
            if success:
                logger.info("✓ Analysis completed successfully")
            else:
                logger.error(f"✗ Analysis failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            logger.error("Analysis timed out after 1 hour")
        except Exception as e:
            logger.error(f"Error running analysis: {e}")
        
        # Record run
        end_time = datetime.now()
        run_record = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': (end_time - start_time).total_seconds(),
            'success': success,
            'timestamp': start_time.strftime('%Y%m%d_%H%M%S')
        }
        
        self.run_history.append(run_record)
        self.last_run = run_record
        
        # Clean up old runs
        self.cleanup_old_runs()
        
        # Send notification
        if self.config['notifications']['enabled']:
            self.send_notification(run_record)
        
        # Save run history
        self.save_run_history()
        
        return success
    
    def cleanup_old_runs(self):
        """Remove old analysis results based on retention policy"""
        retention = self.config['retention']
        analysis_dir = Path('./production_analysis')
        
        if not analysis_dir.exists():
            return
        
        run_dirs = [d for d in analysis_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        
        # Sort by modification time
        run_dirs.sort(key=lambda d: d.stat().st_mtime)
        
        # Remove old runs by count
        if len(run_dirs) > retention['max_runs']:
            for dir_to_remove in run_dirs[:-retention['max_runs']]:
                try:
                    import shutil
                    shutil.rmtree(dir_to_remove)
                    logger.info(f"Removed old run: {dir_to_remove.name}")
                except Exception as e:
                    logger.error(f"Error removing {dir_to_remove}: {e}")
        
        # Remove old runs by age
        cutoff_date = datetime.now() - timedelta(days=retention['keep_days'])
        for run_dir in run_dirs:
            mod_time = datetime.fromtimestamp(run_dir.stat().st_mtime)
            if mod_time < cutoff_date:
                try:
                    import shutil
                    shutil.rmtree(run_dir)
                    logger.info(f"Removed old run: {run_dir.name} (older than {retention['keep_days']} days)")
                except Exception as e:
                    logger.error(f"Error removing {run_dir}: {e}")
    
    def send_notification(self, run_record: Dict):
        """Send email notification about analysis completion"""
        if not self.config['notifications']['enabled']:
            return
        
        try:
            notif = self.config['notifications']
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = notif['smtp_user']
            msg['To'] = notif['email']
            msg['Subject'] = f"Consolidation Analysis - {'Success' if run_record['success'] else 'Failed'}"
            
            # Body
            body = f"""
Consolidation Analysis Completed

Status: {'✓ Successful' if run_record['success'] else '✗ Failed'}
Start Time: {run_record['start_time']}
End Time: {run_record['end_time']}
Duration: {run_record['duration']:.1f} seconds

View results at: production_analysis/run_{run_record['timestamp']}/
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(notif['smtp_server'], notif['smtp_port']) as server:
                server.starttls()
                server.login(notif['smtp_user'], notif['smtp_password'])
                server.send_message(msg)
            
            logger.info("Notification sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def save_run_history(self):
        """Save run history to file"""
        history_file = Path('run_history.json')
        try:
            with open(history_file, 'w') as f:
                json.dump(self.run_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving run history: {e}")
    
    def load_run_history(self):
        """Load run history from file"""
        history_file = Path('run_history.json')
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.run_history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading run history: {e}")
    
    def schedule_runs(self):
        """Set up scheduled runs based on configuration"""
        schedule_config = self.config['schedule']
        mode = schedule_config['mode']
        
        logger.info(f"Scheduling mode: {mode}")
        
        if mode == 'daily':
            schedule.every().day.at(schedule_config['daily_time']).do(self.run_analysis)
            logger.info(f"Scheduled daily run at {schedule_config['daily_time']}")
            
        elif mode == 'weekly':
            day = schedule_config['weekly_day']
            time_str = schedule_config.get('daily_time', '02:00')
            getattr(schedule.every(), day).at(time_str).do(self.run_analysis)
            logger.info(f"Scheduled weekly run on {day} at {time_str}")
            
        elif mode == 'monthly':
            # Monthly scheduling requires custom logic
            logger.info(f"Monthly scheduling on day {schedule_config['monthly_day']}")
            schedule.every().day.at(schedule_config.get('daily_time', '02:00')).do(self.check_monthly_run)
            
        else:
            logger.info("Manual mode - no automatic scheduling")
    
    def check_monthly_run(self):
        """Check if today is the scheduled monthly run day"""
        if datetime.now().day == self.config['schedule']['monthly_day']:
            self.run_analysis()
    
    def start(self):
        """Start the scheduler"""
        logger.info("="*60)
        logger.info("ANALYSIS SCHEDULER STARTED")
        logger.info("="*60)
        
        # Load history
        self.load_run_history()
        
        # Set up schedules
        self.schedule_runs()
        
        # Display next run time
        next_run = schedule.next_run()
        if next_run:
            logger.info(f"Next scheduled run: {next_run}")
        
        logger.info("Scheduler is running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\nScheduler stopped by user")
            self.save_run_history()
    
    def run_once(self):
        """Run analysis once immediately"""
        logger.info("Running analysis once...")
        success = self.run_analysis()
        return success
    
    def status(self):
        """Display scheduler status"""
        print("="*60)
        print("SCHEDULER STATUS")
        print("="*60)
        
        print(f"Mode: {self.config['schedule']['mode']}")
        
        if self.last_run:
            print(f"\nLast Run:")
            print(f"  Time: {self.last_run['start_time']}")
            print(f"  Duration: {self.last_run['duration']:.1f} seconds")
            print(f"  Status: {'Success' if self.last_run['success'] else 'Failed'}")
        
        print(f"\nTotal Runs: {len(self.run_history)}")
        
        if self.run_history:
            successful = sum(1 for r in self.run_history if r['success'])
            print(f"Successful: {successful}/{len(self.run_history)}")
        
        next_run = schedule.next_run()
        if next_run:
            print(f"\nNext Scheduled Run: {next_run}")
        
        print(f"\nRetention Policy:")
        print(f"  Keep for: {self.config['retention']['keep_days']} days")
        print(f"  Max runs: {self.config['retention']['max_runs']}")


def main():
    """Main entry point for scheduler"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Schedule consolidation analysis runs')
    parser.add_argument('--start', action='store_true', help='Start the scheduler')
    parser.add_argument('--once', action='store_true', help='Run analysis once')
    parser.add_argument('--status', action='store_true', help='Show scheduler status')
    parser.add_argument('--config', type=str, help='Path to config file', default='scheduler_config.json')
    
    args = parser.parse_args()
    
    scheduler = AnalysisScheduler(config_file=args.config)
    
    if args.once:
        success = scheduler.run_once()
        exit(0 if success else 1)
    elif args.status:
        scheduler.status()
    elif args.start:
        scheduler.start()
    else:
        print("Use --start to begin scheduling, --once to run immediately, or --status to check status")


if __name__ == "__main__":
    main()