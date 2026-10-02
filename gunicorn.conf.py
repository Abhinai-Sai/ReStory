import os

# Gunicorn configuration for Render deployment
# Increase worker timeout to 300 seconds (5 mins) to prevent 502 WORKER TIMEOUT during CPU inference/model downloads
timeout = 300
workers = 1
threads = 2
keepalive = 65
accesslog = '-'
errorlog = '-'
loglevel = 'info'
