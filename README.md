# Support repo for my gen_ai and agentic_ai work


### Setup

- please make sure script is executable
```bash
chmod +x scripts/dev.py
```

- To add dev dependencies

```commandline
uv add --dev nbstripout nbconvert ruff pyright bandit detect-secrets pre-commit safety
```

- To setup pre-commit hook
```commandline
uv run pre-commit install
```

- Auto update pre-commit config file
```commandline
uv run pre-commit autoupdate
```

- create baseline files
```commandline
uv run detect-secrets scan --all-files --exclude-files '\.git/.*|\.venv/.*|node_modules/.*|\.ruff_cache/.*' > .secrets.baseline
```

- create bandit baseline file
```commandline
uv run bandit -r . -f json -o bandit-report.json
```

- audit secret baseline file
```commandline
uv run detect-secrets audit .secrets.baseline
```




### important commands

- To update pre-commit automatically
```commandline
uv run pre-commit autoupdate
```

- To validate pre-commit config
```commandline
uv run pre-commit validate-config .pre-commit-config.yaml
```

- To test github action in local environment using `docker` and `act` tool
```commandline
act  --container-architecture linux/amd64
```
