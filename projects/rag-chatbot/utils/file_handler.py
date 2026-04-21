import os

DATA_PATH = "data"

def save_uploaded_files(uploaded_files):
    os.makedirs(DATA_PATH, exist_ok=True)

    # delete old files
    for f in os.listdir(DATA_PATH):
        os.remove(os.path.join(DATA_PATH, f))

    for file in uploaded_files:
        path = os.path.join(DATA_PATH, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())