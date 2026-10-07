@echo off
cd /d "%~dp0"
echo Stopping GraphRAG containers (graph data and model files are kept)...
docker compose --profile local down
echo Done.
