"""
Environment Setup Script
Installs all required dependencies for the analysis system
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages"""
    
    print("="*60)
    print("SETTING UP ENVIRONMENT")
    print("="*60)
    
    # Core packages
    packages = [
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'scipy>=1.9.0',
        'google-cloud-storage>=2.10.0',
        'google-auth>=2.20.0',
        'matplotlib>=3.6.0',
        'seaborn>=0.12.0',
        'plotly>=5.14.0',
        'scikit-learn>=1.2.0',
        'openpyxl>=3.1.0',  # For Excel files
        'jinja2>=3.1.0',    # For HTML templates
        'markdown>=3.4.0',  # For markdown reports
        'python-dateutil>=2.8.0',
        'pytz>=2023.3',
        'schedule',  # For scheduling
    ]
    
    print("\nInstalling packages...")
    print("-"*40)
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.STDOUT)
            print(f"  ✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"  ✗ Failed to install {package}")
            # Try without version specification
            try:
                base_package = package.split('>=')[0].split('==')[0]
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', base_package],
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.STDOUT)
                print(f"  ✓ {base_package} installed (latest version)")
            except:
                print(f"  ✗ Could not install {base_package}")
    
    print("\n" + "="*60)
    print("VERIFYING INSTALLATION")
    print("="*60)
    
    # Verify critical imports
    critical_modules = {
        'pandas': 'pd',
        'numpy': 'np',
        'google.cloud.storage': 'storage',
        'openpyxl': 'openpyxl',
        'matplotlib': 'plt',
        'plotly': 'plotly'
    }
    
    all_good = True
    for module, alias in critical_modules.items():
        try:
            if '.' in module:
                parts = module.split('.')
                exec(f"from {'.'.join(parts[:-1])} import {parts[-1]} as {alias}")
            else:
                exec(f"import {module} as {alias}")
            print(f"✓ {module} is working")
        except ImportError as e:
            print(f"✗ {module} failed to import: {e}")
            all_good = False
    
    if all_good:
        print("\n✅ All packages installed successfully!")
    else:
        print("\n⚠️ Some packages failed. You may need to install them manually.")
    
    return all_good


def main():
    """Main setup function"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║          CONSOLIDATION ANALYSIS ENVIRONMENT SETUP           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    if sys.version_info < (3, 7):
        print("⚠️ Warning: Python 3.7+ is recommended")
    
    # Install packages
    success = install_packages()
    
    if success:
        print("\n" + "="*60)
        print("SETUP COMPLETE")
        print("="*60)
        print("\nYou can now run the analysis pipeline:")
        print("  python run_full_pipeline.py --test")
        print("\nOr for quick analysis:")
        print("  python run_full_pipeline.py --quick")
    else:
        print("\n" + "="*60)
        print("SETUP INCOMPLETE")
        print("="*60)
        print("\nPlease install missing packages manually:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()