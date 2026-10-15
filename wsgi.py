"""WSGI entry point: `gunicorn wsgi:app`."""
from nexusai.app import create_app

app = create_app()
