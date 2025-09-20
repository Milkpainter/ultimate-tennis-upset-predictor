#!/usr/bin/env python3
"""
Ultimate American Football Upset Predictor - Setup Script

Automated setup script to initialize the football prediction environment.
Handles dependencies, data collection, and initial model training.

Author: Milkpainter
Version: 1.0
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """
    Print setup banner.
    """
    print("🏈" * 20)
    print("🏈 ULTIMATE AMERICAN FOOTBALL UPSET PREDICTOR SETUP 🏈")
    print("🏈" * 20)
    print()
    print("This script will set up your environment for:")
    print("• Advanced NFL/College Football prediction")
    print("• Ensemble machine learning models")
    print("• Genetic algorithm feature engineering")
    print("• Upset detection and value betting")
    print("• Kelly Criterion position sizing")
    print()
    print("Target Performance:")
    print("• >82% overall accuracy")
    print("• >65% upset detection rate")
    print("• >15% annual ROI")
    print()

def check_python_version():
    """
    Check if Python version is compatible.
    """
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """
    Install required packages.
    """
    print("\n📦 Installing requirements...")
    
    requirements_file = "requirements_football.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    try:
        # Upgrade pip first
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        
        print("✅ Requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """
    Create necessary project directories.
    """
    print("\n📁 Creating project directories...")
    
    directories = [
        "data/nfl",
        "data/college", 
        "data/cache",
        "data/odds",
        "data/weather",
        "models/saved",
        "models/checkpoints",
        "results",
        "logs",
        "plots",
        "config",
        "tests",
        "docs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    return True

def create_config_files():
    """
    Create configuration templates.
    """
    print("\n⚙️ Creating configuration files...")
    
    # Create main config
    config_content = """
# Ultimate American Football Upset Predictor Configuration

# Data Collection
SEASONS_TO_COLLECT = [2021, 2022, 2023, 2024]
WEATHER_API_KEY = "your_openweathermap_api_key_here"
ODDS_API_KEY = "your_odds_api_key_here"

# Model Parameters
ESTIMATOR_WEIGHTS = {
    "xgboost": 0.40,
    "random_forest": 0.25,
    "catboost": 0.20,
    "lstm": 0.15
}

# Genetic Algorithm
GA_POPULATION_SIZE = 50
GA_GENERATIONS = 100
GA_MUTATION_RATE = 0.1
GA_CROSSOVER_RATE = 0.8

# Upset Detection
BANKROLL = 10000
MAX_BET_PERCENTAGE = 0.05
MIN_EDGE = 0.05
MIN_CONFIDENCE = 0.55
KELLY_FRACTION = 0.25

# Performance Targets
TARGET_ACCURACY = 0.82
TARGET_UPSET_DETECTION = 0.65
TARGET_ROI = 0.15
"""
    
    with open("config/settings.py", "w") as f:
        f.write(config_content)
    
    # Create logging config
    logging_config = """
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/football_predictor_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('football_predictor')
"""
    
    with open("config/logging_setup.py", "w") as f:
        f.write(logging_config)
    
    print("✅ Configuration files created")
    return True

def test_installation():
    """
    Test if key packages are working.
    """
    print("\n🧪 Testing installation...")
    
    test_imports = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('sklearn', None),
        ('xgboost', 'xgb'),
        ('catboost', None),
        ('tensorflow', 'tf'),
        ('nfl_data_py', 'nfl')
    ]
    
    failed_imports = []
    
    for package, alias in test_imports:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️ Failed imports: {failed_imports}")
        print("Try running: pip install -r requirements_football.txt")
        return False
    
    print("\n✅ All core packages imported successfully!")
    return True

def create_quick_start_script():
    """
    Create a quick start script for new users.
    """
    print("\n🚀 Creating quick start script...")
    
    quick_start = """
#!/usr/bin/env python3
\"\"\"
Quick Start - Ultimate American Football Upset Predictor

Run this script to get started with football prediction.
\"\"\"

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_training_pipeline import main
import argparse

def quick_start():
    print("🏈 QUICK START - American Football Upset Predictor")
    print("=" * 50)
    
    # Create args for quick demo
    class QuickArgs:
        seasons = [2023, 2024]  # Just recent seasons for quick start
        weather_api_key = None  # No weather data for quick start
        quick_mode = True
    
    args = QuickArgs()
    
    print("Starting quick training with recent seasons (2023-2024)...")
    print("This will take 10-15 minutes to complete.")
    print()
    
    # Run main pipeline
    return main(args)

if __name__ == "__main__":
    exit_code = quick_start()
    
    if exit_code == 0:
        print("\n🎉 QUICK START COMPLETE!")
        print("\nNext steps:")
        print("1. Review results in ./results/")
        print("2. Check model performance metrics")
        print("3. Run full training with all seasons:")
        print("   python main_training_pipeline.py --seasons 2020 2021 2022 2023 2024")
        print("4. Add weather data with API key:")
        print("   python main_training_pipeline.py --weather-api-key YOUR_KEY")
    else:
        print("\n❌ Quick start failed. Check error messages above.")
    
    sys.exit(exit_code)
"""
    
    with open("quick_start.py", "w") as f:
        f.write(quick_start)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("quick_start.py", 0o755)
    
    print("✅ Quick start script created: quick_start.py")
    return True

def create_readme_update():
    """
    Create a getting started guide.
    """
    print("\n📝 Creating getting started guide...")
    
    getting_started = """
# 🏈 Getting Started - American Football Upset Predictor

## Quick Start (5 minutes)

```bash
# 1. Run setup (if not done already)
python setup_football_predictor.py

# 2. Quick training demo
python quick_start.py
```

## Full Training Pipeline

```bash
# Train with all available seasons
python main_training_pipeline.py --seasons 2020 2021 2022 2023 2024

# Add weather data (requires API key)
python main_training_pipeline.py --weather-api-key YOUR_OPENWEATHERMAP_KEY
```

## Project Structure

```
american-football-predictor/
├── models/
│   ├── ensemble/              # Ensemble prediction models
│   └── saved/                 # Trained model files
├── data/
│   ├── collection/            # Data collection scripts
│   ├── nfl/                   # NFL datasets
│   └── college/               # College datasets
├── feature_engineering/
│   └── genetic_algorithm.py   # GA feature optimization
├── upset_detection/
│   └── upset_detector.py      # Upset identification
├── main_training_pipeline.py  # Main training script
├── quick_start.py             # Quick demo
└── requirements_football.txt  # Dependencies
```

## Key Components

### 1. Data Collection
- **NFL Data**: nflfastR, ESPN API, Pro Football Reference
- **College Data**: CFBD API, Sports Reference  
- **Market Data**: Betting odds and line movement
- **Weather**: OpenWeatherMap for outdoor games

### 2. Feature Engineering
- **Genetic Algorithm**: Automated feature selection
- **Advanced Metrics**: EPA, DVOA, efficiency ratings
- **Time Weighting**: Recent performance emphasis
- **Situational**: Weather, injuries, motivation

### 3. Ensemble Models
- **XGBoost** (40%): Primary algorithm
- **Random Forest** (25%): Robustness
- **CatBoost** (20%): Categorical handling
- **LSTM** (15%): Sequential patterns

### 4. Upset Detection
- **Logic**: Model >50% but Market <50%
- **Kelly Criterion**: Optimal bet sizing
- **Risk Management**: 5% max bet, stop losses
- **Expected Value**: ROI calculations

## Performance Targets

| Metric | Target | Best Research Results |
|--------|--------|-----------------------|
| Overall Accuracy | >82% | 85% (top studies) |
| Upset Detection | >65% | 75% (optimistic) |
| Annual ROI | >15% | 25% (disciplined Kelly) |
| Sharpe Ratio | >1.5 | 3.0+ (best case) |

## Configuration

Edit `config/settings.py` to customize:
- Seasons to collect
- API keys
- Model weights
- Risk parameters
- Performance targets

## Results

After training, check:
- `results/training_results.json` - Performance metrics
- `results/feature_importance.csv` - Selected features
- `models/saved/` - Trained model files
- `logs/` - Training logs

## Next Steps

1. **Live Prediction**: Use trained models for current games
2. **Paper Trading**: Test strategies without real money
3. **API Integration**: Connect to betting exchanges
4. **Monitoring**: Track real-world performance

## Support

For issues or questions:
1. Check `logs/` for error details
2. Review configuration in `config/`
3. Verify all requirements are installed
4. Test with `quick_start.py` first

---

**Remember**: This is for educational purposes. Always bet responsibly!
"""
    
    with open("GETTING_STARTED.md", "w") as f:
        f.write(getting_started)
    
    print("✅ Getting started guide created: GETTING_STARTED.md")
    return True

def main():
    """
    Main setup function.
    """
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️ Installation failed. You may need to install packages manually.")
        print("Try: pip install -r requirements_football.txt")
    
    # Create directories
    if not create_directories():
        return 1
    
    # Create config files
    if not create_config_files():
        return 1
    
    # Test installation
    if not test_installation():
        print("\n⚠️ Some packages failed to import. Check error messages above.")
    
    # Create quick start script
    if not create_quick_start_script():
        return 1
    
    # Create documentation
    if not create_readme_update():
        return 1
    
    # Final message
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("🚀 Ready to predict football upsets!")
    print()
    print("Next steps:")
    print("1. Quick demo:    python quick_start.py")
    print("2. Full training: python main_training_pipeline.py")
    print("3. Read guide:    cat GETTING_STARTED.md")
    print()
    print("Configuration:")
    print("- Edit config/settings.py for your preferences")
    print("- Add API keys for weather and odds data")
    print("- Adjust model parameters and risk settings")
    print()
    print("⚠️  Remember: This is for educational purposes!")
    print("   Always bet responsibly and within your means.")
    print()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
