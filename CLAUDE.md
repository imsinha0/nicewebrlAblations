# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains ablation studies for NiceWebRL, a Python library for creating interactive reinforcement learning web applications using NiceGUI. The project includes multiple experimental variations and two main environments: Crafter and XLand-MiniGrid.

## Project Structure

- `currentNiceWebRL/` - Core NiceWebRL library implementation
- `crafter/` - Crafter environment web application
- `xlandminigrid/` - XLand-MiniGrid environment web application
- `ablation1/`, `ablation2/`, `ablation3/` - Different experimental ablations
- Each environment has its own `web_app.py`, `experiment_structure.py`, and configuration files

## Development Commands

### Running Applications

**Crafter Environment:**
```bash
cd crafter
python web_app.py
```

**XLand-MiniGrid Environment:**
```bash
cd xlandminigrid
python web_app.py
```

### Package Management

**For xlandminigrid (uses uv):**
```bash
cd xlandminigrid
uv sync --frozen  # Install dependencies
uv run python web_app.py  # Run with uv
```

**For crafter (uses pip):**
```bash
cd crafter
pip install -r requirements.txt
```

### Code Formatting (xlandminigrid only)

```bash
cd xlandminigrid
black .  # Format code
isort .  # Sort imports
pytest  # Run tests (if available)
```

### Docker Deployment

Both environments have Dockerfiles for deployment:

**Crafter:**
```bash
docker build -t crafter-app .
docker run crafter-app
```

**XLand-MiniGrid:**
```bash
docker build -t xlandminigrid-app .
docker run xlandminigrid-app
```

## Architecture

### Core Components

- **NiceWebRL Core (`currentNiceWebRL/`)**: Main library with experiment management, stages, JAX integration, logging, and utilities
- **Ablation System**: Environment variable `ABLATION_MODE` controls which implementation variant to use
- **Experiment Structure**: Modular experiment definitions with stages and blocks
- **JAX Integration**: Random number generation, serialization, and web environment interface via `nicejax.py`

### Key Modules

- `stages.py` - Experiment stage management and execution
- `experiment.py` - High-level experiment container and coordination
- `nicejax.py` - JAX integration for web-based RL environments
- `logging.py` - Structured logging with user session tracking
- `utils.py` - Utilities for UI, data handling, and user management

### Ablation Mode System

The codebase uses `ABLATION_MODE` environment variable to switch between different implementations:
- `normal` - Default currentNiceWebRL implementation
- `ablation1` - Uses `ablation1/ablation1nicejax.py`
- `ablation2` - Uses `ablation2/ablation2stages.py`
- `ablation3` - Uses `ablation3/ablation3nicejax.py`
- `ablation4` - Combines ablation1 and ablation2

### Database and Storage

- Uses Tortoise ORM with SQLite backend (`db.sqlite`)
- User session data stored with NiceGUI's storage system
- Google Cloud Storage integration for data persistence
- MessagePack format for efficient data serialization

## Dependencies

- **Python 3.10+** required
- **Core**: NiceGUI, FastAPI, Tortoise ORM, JAX/JAXlib
- **RL**: Environment-specific packages (craftax, xminigrid)
- **Data**: Polars, NumPy, Pillow for data processing and visualization
- **Cloud**: Google Cloud Storage for data persistence