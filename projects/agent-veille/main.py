from dotenv import load_dotenv

from graph import build

load_dotenv()

if __name__ == "__main__":
    build().invoke({})
