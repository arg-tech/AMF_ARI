#!/bin/sh
source venv/bin/activate
exec gunicorn -b :5001 --access-logfile - --error-logfile - app --timeout 300
