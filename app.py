"""
Main Flask application for ML Service
Run this file to start the Flask API server
Untuk production, gunicorn dijalankan langsung via Procfile: gunicorn flask_api:app
"""

import os
from flask_api import app  # Model sudah di-init otomatis saat flask_api di-import

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print("[OK] Starting Flask dev server")
    app.run(debug=debug_mode, port=port, host='0.0.0.0')
