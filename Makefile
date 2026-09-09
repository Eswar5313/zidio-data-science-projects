.PHONY: all pipeline forecast risk eda web app api test
all: ; python src/run_all.py
pipeline: ; python src/pipeline.py
forecast: ; python src/forecast.py
risk: ; python src/risk.py
eda: ; python src/eda.py
web: ; python src/export_web.py
app: ; streamlit run app/streamlit_app.py
api: ; uvicorn service.main:app --reload --port 8000
test: ; python -m pytest tests -q
