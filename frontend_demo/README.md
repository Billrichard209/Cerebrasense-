# Frontend Demo: CerebraSense MRI Decision Support

A polished single-page frontend prototype for demonstrating a modern UI direction for the project.

## Run locally

### Easiest option on Windows

Double-click `run_frontend_demo_localhost.cmd` 


That launcher will:

- find Python automatically
- start a local static server for this folder
- open the demo in your browser at `http://127.0.0.1:8080/`

Press `Ctrl+C` in the terminal window when you want to stop the server.

### Manual option

From the project root:

```bash
cd frontend_demo
python -m http.server 8080 --bind 127.0.0.1
```

Then visit: `http://127.0.0.1:8080/`
