from dotenv import load_dotenv

load_dotenv()

from src.ui.app import run_app


def main():
    run_app()


if __name__ == "__main__":
    main()
