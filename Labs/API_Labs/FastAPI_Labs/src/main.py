from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class CaliforniaData(BaseModel):
    median_income: float
    median_house_age: float
    average_rooms: float
    average_bedrooms: float
    population: float
    average_occupancy: float
    latitude: float
    longitude: float

class CaliforniaResponse(BaseModel):
    response:float

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=CaliforniaResponse)
async def predict_california(california_features: CaliforniaData):
    try:
        features = [[california_features.median_income, california_features.median_house_age,
                    california_features.average_rooms, california_features.average_bedrooms,
                    california_features.population, california_features.average_occupancy,
                    california_features.latitude, california_features.longitude]]

        prediction = predict_data(features)
        return CaliforniaResponse(response=float(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
