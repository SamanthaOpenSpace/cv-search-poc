# cv-search

Step 0: Repo scaffold & env only.

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# 1) init DB (if not done yet)
python .\main.py init-db 

# 2) ingest mock CVs // both local db + vector store(upsert)
python .\main.py ingest-mock 

# 3) parse-request
python .\main.py parse-request

# 4) search-seat
python .\main.py search-seat

# 5) search hybrid by default
python main.py search-seat --criteria ./criteria.json --topk 2

## RUN UI
streamlit run app.py

## PRESALE

# 1) Presale — roles only (budget ignored)
python main.py presale-plan --text "Mobile + web app with Flutter/React; AI chatbot for goal setting; partner marks failures; donation on failure."

# 2) Project phase — free text → seats → per-seat shortlists
python main.py project-search --db ./cvsearch.db --text "Mobile+web app in Flutter/React; AI chatbot; partner marks failures; donation on failure." --topk 3

# 3) Or, with explicit canonical criteria (JSON)
python main.py project-search --criteria ./criteria.json --topk 3



