# Creates a venv, installs dependencies, runs the script

# Create virtual environment if not exists
if (-Not (Test-Path ".\MorphOriginVenv")) {
    Write-Host "Creating virtual environment..."
    python -m venv MorphOriginVenv
}

# Activate virtual environment
$envPath = ".\MorphOriginVenv\Scripts\Activate.ps1"
if (Test-Path $envPath) {
    Write-Host "Activating virtual environment..."
    & $envPath
} else {
    Write-Error "Cannot activate virtual environment. Activation script not found."
    exit 1
}

# Upgrade pip and install requirements
Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements_win.txt

# Run main script with default setting
Write-Host "Running the main script..."
python src\main.py --enable_all