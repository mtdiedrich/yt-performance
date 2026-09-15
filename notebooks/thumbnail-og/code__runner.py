import subprocess
import time
import signal

# Path to the script to be run
script_path = 'scrape.py'

# Time (in seconds) to wait before restarting
interval_seconds = 3600  # 1 hour

while True:
    print("Starting scrape.py script...")
    
    # Start scrape.py in a subprocess
    process = subprocess.Popen(['python', script_path])

    # Wait for the specified interval
    time.sleep(interval_seconds)

    # Terminate the process after one hour
    print("Stopping scrape.py script...")
    process.send_signal(signal.SIGTERM)  # Gracefully terminate the process

    # Wait for the process to terminate completely
    process.wait()
    print("scrape.py script terminated. Restarting...")

    # Optional: Short delay before restarting
    time.sleep(5)  # Give a small gap before restarting
