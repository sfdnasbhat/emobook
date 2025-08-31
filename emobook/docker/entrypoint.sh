#!/bin/sh
set -e

# Ensure all needed dirs exist (volumes might be empty first run)
mkdir -p /app/emobook/data/raw \
         /app/emobook/data/clean \
         /app/emobook/data/chunks \
         /app/emobook/data/scored \
         /app/emobook/uploads

# Take ownership (volumes default to root)
chown -R appuser:appuser /app/emobook

# Drop privileges and start the app
exec su -s /bin/sh -c "python /app/app_gradio.py" appuser
