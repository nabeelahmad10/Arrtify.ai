from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from MultilingualTextToImageGenerator import generate_image  # Import your ML model function

app = FastAPI()

# Mount the static directories for CSS and images
app.mount("/styles", StaticFiles(directory="styles"), name="styles")
app.mount("/images", StaticFiles(directory="Images"), name="images")

# Root endpoint to serve the index.html
@app.get("/", response_class=HTMLResponse)
async def read_index():
    html_file = Path("html/index.html")
    return html_file.read_text()

# About endpoint
@app.get("/about", response_class=HTMLResponse)
async def read_about():
    html_file = Path("html/about.html")
    return html_file.read_text()

# Generate endpoint
@app.get("/generate", response_class=HTMLResponse)
async def read_generate():
    html_file = Path("html/generate.html")
    return html_file.read_text()

# API endpoint to generate image
@app.post("/generate_image")
async def generate_image_endpoint(prompt: str = Form(...)):
    image_path = generate_image(prompt)  # Use your model to generate the image
    return FileResponse(image_path)

if __name__ == "__main__":
    # Run the app with Uvicorn
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
