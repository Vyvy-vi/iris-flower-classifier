from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from typing import Optional,  Annotated
from classification_model import predict

app = FastAPI()
app.mount("/static/", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


class InputParams(BaseModel):
    petal_length: Optional[str] = None
    petal_width: Optional[str] = None
    sepal_length: Optional[str] = None
    sepal_width: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "params": ["petal_length", "petal_width", "sepal_length", "sepal_width"]
    })


@app.get("/valid-float", response_class=HTMLResponse)
async def validate_floats(request: Request):
    param, val = list(request.query_params.items())[0]
    if not val.isalpha():
        return templates.TemplateResponse("param-field.html", {"request": request, "param": param, "error": None, "value": val}) 
    else:
        return templates.TemplateResponse("param-field.html", {"request": request, "param": param, "error": "This param must be a float64 value"})  


@app.post("/classify", response_class=HTMLResponse)
async def classify_endpoint(
    request: Request,
    petal_length: Annotated[str, Form()],
    petal_width: Annotated[str, Form()],
    sepal_length: Annotated[str, Form()],
    sepal_width: Annotated[str, Form()]
):
    predicted_class = predict([petal_length, petal_width, sepal_length, sepal_width])
    print(predicted_class)
    return templates.TemplateResponse("classification.html", {"request": request, "class": predicted_class})
