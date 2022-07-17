from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"API TFM": "Author Gorka Zabarte Moreno"}
