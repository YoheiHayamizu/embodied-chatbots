# Embodied Chatbots

This repository contains the code that implements the embodied chatbot system. The system is designed to enable chatbots to interact with the physical world through various sensors and actuators. The main components of the system include:
- Sensor integration: The system can integrate with various sensors to gather information about the environment, such as cameras, microphones, and other IoT devices.
- Natural language processing: The system uses natural language processing techniques to understand and generate human-like responses based on the input from the sensors and the context of the conversation.
- Dialogue management: The system manages the flow of the conversation, ensuring that the chatbot can maintain context and provide relevant responses.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

1. Install uv (first time only):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies (creates `.venv` automatically):
```bash
uv sync
```

## Running the code

Use `uv run` to execute scripts inside the project environment:

```bash
uv run python main.py
```

Or activate the virtualenv manually:

```bash
source .venv/bin/activate
python main.py
```

## Managing dependencies

```bash
uv add <package>           # add a runtime dependency
uv add --dev <package>     # add a dev dependency
uv remove <package>        # remove a dependency
uv lock --upgrade          # upgrade locked versions
```

