#!/usr/bin/env python
"""
Setup script to initialize the E-commerce Analytics Platform.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger
from src.utils.database import init_db


logger = get_logger("setup")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup E-commerce Analytics Platform")
    parser.add_argument("--init-db", action="store_true", help="Initialize database")
    parser.add_argument("--create-env", action="store_true", help="Create .env file from .env.example")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    
    return parser.parse_args()


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_example_path = root_dir / ".env.example"
    env_path = root_dir / ".env"
    
    if not env_example_path.exists():
        logger.error(f".env.example file not found at {env_example_path}")
        return False
    
    if env_path.exists():
        logger.info(f".env file already exists at {env_path}")
        return True
    
    try:
        with open(env_example_path, "r") as example_file:
            content = example_file.read()
        
        with open(env_path, "w") as env_file:
            env_file.write(content)
        
        logger.info(f"Created .env file at {env_path}")
        logger.info("Please update the .env file with your credentials")
        return True
    except Exception as e:
        logger.error(f"Failed to create .env file: {str(e)}")
        return False


def install_dependencies():
    """Install Python dependencies."""
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    requirements_path = root_dir / "requirements.txt"
    
    if not requirements_path.exists():
        logger.error(f"requirements.txt not found at {requirements_path}")
        return False
    
    try:
        logger.info("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], check=True)
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {str(e)}")
        return False


def initialize_database():
    """Initialize the database."""
    try:
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return False


def create_directories():
    """Create necessary directories if they don't exist."""
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    directories = [
        root_dir / "logs",
        root_dir / "data" / "raw",
        root_dir / "data" / "processed",
        root_dir / "data" / "interim",
        root_dir / "models",
        root_dir / "artifacts",
    ]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    return True


def main():
    """Main setup function."""
    args = parse_args()
    
    # Determine which steps to run
    run_all = args.all
    run_init_db = args.init_db or run_all
    run_create_env = args.create_env or run_all
    run_install_deps = args.install_deps or run_all
    
    # Always create directories
    create_directories()
    
    # Run selected steps
    if run_create_env:
        create_env_file()
    
    if run_install_deps:
        install_dependencies()
    
    if run_init_db:
        initialize_database()
    
    logger.info("Setup completed")


if __name__ == "__main__":
    main() 