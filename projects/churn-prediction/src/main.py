from fastapi import FastAPI, HTTPException
import pandas as pd
from schemas import CustomerInput, CustomerOutput
from business import run_business_pipeline

app = FastAPI(
    title="Churn Prediction API",
    description="Predict churn risk and assign retention actions",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=CustomerOutput)
def predict(customer: CustomerInput):
    try:
        df = pd.DataFrame([customer.model_dump()])
        result = run_business_pipeline(df)
        row = result.iloc[0]
        return CustomerOutput(
            customerID=customer.customerID,
            SegmentName=row["SegmentName"],
            churn_proba=round(float(row["churn_proba"]), 4),
            risk=row["risk"],
            action=row["action"],
            priority_score=round(float(row["priority_score"]), 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=list[CustomerOutput])
def predict_batch(customers: list[CustomerInput]):
    try:
        df = pd.DataFrame([c.model_dump() for c in customers])
        result = run_business_pipeline(df)
        return [
            CustomerOutput(
                customerID=customers[i].customerID,
                SegmentName=row["SegmentName"],
                churn_proba=round(float(row["churn_proba"]), 4),
                risk=row["risk"],
                action=row["action"],
                priority_score=round(float(row["priority_score"]), 4)
            )
            for i, row in result.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))