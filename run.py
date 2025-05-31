import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    print("🧠 Mental Health Counselor Assistant")
    print("=" * 40)
    
    print("Starting application...")
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main() 