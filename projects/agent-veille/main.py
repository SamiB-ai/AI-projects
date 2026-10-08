from dotenv import load_dotenv
 
load_dotenv()
 
from graph import build
 
if __name__ == "__main__":
    build().invoke({})
 