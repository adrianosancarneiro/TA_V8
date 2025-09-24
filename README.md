# Team Agent Platform V8 (TA_V8)

A comprehensive multi-agent orchestration and execution platform designed for high-performance AI workloads with RTX 5090 optimization.

## Overview

The Team Agent Platform V8 consists of three main components:

- **TAB_V1** (Team Agent Builder) - Design-time agent team configuration
- **TAE_V1** (Team Agent Executor) - Runtime agent task execution  
- **TAO_V1** (Team Agent Orchestrator) - Multi-agent workflow coordination

## Features

- 🤖 Multi-agent team orchestration with LangGraph
- 🚀 RTX 5090 optimized Docker containers
- 🔍 Semantic tool recommendation with Qdrant
- 🗄️ Multi-database support (PostgreSQL, Neo4j, Qdrant)
- 🐳 Complete containerized deployment stack
- ⚡ High-performance GPU acceleration
- 🔧 UV package manager integration

## Quick Start

### Prerequisites

- Python 3.11+
- UV package manager
- Docker with NVIDIA Container Toolkit
- RTX 5090 or compatible GPU

### Installation with UV

```bash
# Clone the repository
git clone https://github.com/adrianosancarneiro/TA_V8.git
cd TA_V8

# Create virtual environment with UV
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies with UV
uv pip install -e .

# Install development dependencies
uv pip install -e .[dev]

# For GPU support
uv pip install -e .[gpu]
```

### Docker Setup

```bash
# Setup high-performance Docker stack
chmod +x AI_Support_Services_Containers/setup-high-performance-stack.sh
./AI_Support_Services_Containers/setup-high-performance-stack.sh
```

## Project Structure

```
TA_V8/
├── TAB_V1/                         # Team Agent Builder module
├── TAE_V1/                         # Team Agent Executor module  
├── TAO_V1/                         # Team Agent Orchestrator module
├── scripts/                        # CLI scripts and utilities
├── AI_Support_Services_Containers/ # Docker configurations
├── version.py                      # Version and package info
├── pyproject.toml                  # UV/Python project configuration
└── RTX 5090 sm_120 CUDA Compatibility Guide for Docker ML Workloads.pdf
```

## Components

### Team Agent Builder (TAB_V1)
- Conversational wizard for agent team creation
- YAML configuration import/validation
- Semantic tool recommendation
- Multi-tenant support

### Team Agent Executor (TAE_V1)
- Runtime agent task execution
- Tool integration and management
- Performance monitoring

### Team Agent Orchestrator (TAO_V1)
- Multi-agent workflow coordination
- Session management
- Integration interfaces

## Docker Services

The platform includes optimized Docker containers for:
- **Ollama**: Local LLM inference with GPU acceleration
- **Neo4j**: Graph database for agent relationships
- **Multilingual E5**: Semantic embedding service

## Development

### Using UV for Development

```bash
# Install development dependencies
uv pip install -e .[dev]

# Run tests
pytest

# Code formatting
black ta_v8/
isort ta_v8/

# Type checking
mypy ta_v8/

# Run pre-commit hooks
pre-commit run --all-files
```

### Health Checks

```bash
# Check system health
ta-health

# Setup services
ta-setup
```

## GPU Optimization

This platform is specifically optimized for RTX 5090 with:
- CUDA 12.3+ compatibility
- SM_120 architecture support
- Memory-optimized container configurations
- High-throughput inference pipelines

## Contributing

1. Fork the repository
2. Create a feature branch with UV: `uv venv feature-branch`
3. Install dependencies: `uv pip install -e .[dev]`
4. Make your changes
5. Run tests and linting
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: https://github.com/adrianosancarneiro/TA_V8/issues
- Documentation: https://github.com/adrianosancarneiro/TA_V8/wiki

---

**Note**: This platform uses UV as the preferred package manager. All dependency management should be done through UV commands rather than pip.