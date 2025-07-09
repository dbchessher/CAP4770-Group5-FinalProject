#!/bin/bash

echo "Detecting operating system..."
OS_TYPE="$(uname -s)"

install_python() {
    echo "Attempting to install Python 3..."

    case "$OS_TYPE" in
        Linux)
            if command -v apt &> /dev/null; then
                sudo apt update
                sudo apt install -y python3 python3-pip python3-venv
            elif command -v dnf &> /dev/null; then
                sudo dnf install -y python3 python3-pip
            else
                echo "Unsupported Linux package manager. Please install Python 3 manually."
                exit 1
            fi
            ;;
        Darwin)
            if ! command -v brew &> /dev/null; then
                echo "Homebrew not found. Please install Homebrew: https://brew.sh"
                exit 1
            fi
            brew install python
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "Detected Windows. Please install Python manually from https://www.python.org and run this script in Git Bash or WSL after."
            exit 1
            ;;
        *)
            echo "Unsupported OS: $OS_TYPE"
            exit 1
            ;;
    esac
}

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found."
    install_python
else
    echo "Python 3 is already installed."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OS_TYPE" == "Darwin" || "$OS_TYPE" == "Linux" ]]; then
    source venv/bin/activate
elif [[ "$OS_TYPE" == MINGW* || "$OS_TYPE" == MSYS* || "$OS_TYPE" == CYGWIN* ]]; then
    source venv/Scripts/activate
else
    echo "Unsupported environment for venv activation."
    exit 1
fi

# Upgrade pip
echo "⬆Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing required Python packages..."
pip install pandas matplotlib seaborn

echo "All dependencies installed successfully."
