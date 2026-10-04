from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from predict import predict_data


app = FastAPI()

WINE_CLASSES = ["cultivar_1", "cultivar_2", "cultivar_3"]

class WineData(BaseModel):
    """
    Pydantic BaseModel representing the chemical analysis of a wine.

    Attributes:
        The 13 measurements from the UCI Wine dataset, in the same order as the
        dataset columns. All values must be greater than 0.
    """
    alcohol: float = Field(gt=0)
    malic_acid: float = Field(gt=0)
    ash: float = Field(gt=0)
    alcalinity_of_ash: float = Field(gt=0)
    magnesium: float = Field(gt=0)
    total_phenols: float = Field(gt=0)
    flavanoids: float = Field(gt=0)
    nonflavanoid_phenols: float = Field(gt=0)
    proanthocyanins: float = Field(gt=0)
    color_intensity: float = Field(gt=0)
    hue: float = Field(gt=0)
    od280_od315_of_diluted_wines: float = Field(gt=0)
    proline: float = Field(gt=0)

    # first wine in the dataset, shown as the default request in /docs
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "alcohol": 14.23,
            "malic_acid": 1.71,
            "ash": 2.43,
            "alcalinity_of_ash": 15.6,
            "magnesium": 127,
            "total_phenols": 2.8,
            "flavanoids": 3.06,
            "nonflavanoid_phenols": 0.28,
            "proanthocyanins": 2.29,
            "color_intensity": 5.64,
            "hue": 1.04,
            "od280_od315_of_diluted_wines": 3.92,
            "proline": 1065
        }]
    })

class WineResponse(BaseModel):
    response: int
    class_name: str
    confidence: float



"""Modern web apps use a technique named routing. This helps the user remember the URLs.
For instance, instead of having /booking.php they see /booking/. Instead of /account.asp?id=1234/
they'd see /account/1234/."""

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Concurrent (multiple tasks can run simultaneously)"""
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    """
    Predict the wine cultivar from its chemical measurements.
    Returns the predicted class, its name, and the model's confidence (0 to 1).
    Raises:
        HTTPException: Returns a 500 status code with error details if prediction fails.
    """
    try:
        features = [[wine_features.alcohol, wine_features.malic_acid, wine_features.ash,
                     wine_features.alcalinity_of_ash, wine_features.magnesium,
                     wine_features.total_phenols, wine_features.flavanoids,
                     wine_features.nonflavanoid_phenols, wine_features.proanthocyanins,
                     wine_features.color_intensity, wine_features.hue,
                     wine_features.od280_od315_of_diluted_wines, wine_features.proline]]

        prediction, probabilities = predict_data(features)
        predicted_class = int(prediction[0])
        confidence = float(probabilities[0][predicted_class])

        return WineResponse(response=predicted_class,
                            class_name=WINE_CLASSES[predicted_class],
                            confidence=round(confidence, 3))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))