Projektbeskrivning: Käbbel-AI
Käbbel-AI är en intelligent analytikerplattform som ger användare möjlighet att söka, visualisera och analysera debatter och partiprogram från den svenska riksdagen (2012–2026). Genom att kombinera modern sökteknik med stora språkmodeller (LLM) förvandlar applikationen tusentals timmar av politiskt "käbbel" till strukturerade insikter och faktabaserade sammanfattningar.

Projektet är uppdelat med 2st olika AI modeller som används i tre steg.
- NR1: Gemini flash 2.0-modell vars uppgift är att dekonstruera frågan. Den agerar som en "router" som bestämmer:
Relevant tidsperiod: Ska vi titta på nuvarande mandatperiod eller historiska data?
Behov av statistik: Är frågan av typen "vem pratar mest om..."?
Källval: Behövs officiella partiprogram (ideologi) eller faktiska debattprotokoll (handling)?

-NR2: paraphrase-multilingual-MiniLM-L12-v2. Den har i uppgift att utefter sökparametrarna som NR1 gav den hitta relevanta dokument i en vektordatabas.

-NR3: Ytterligare en Gemini flash 2.0 instans denna har i uppgift att utefter datan som NR2 gav den, sammanfatta och formulera ett svar på frågan från användaren.

Hur man startar projektet.
    1. Skapa en .env fil i roten av projektet. I denna skriv GEMINI_API_KEY=[Med_din_gemini_api_nyckel].
    2. Skapa en python venv och kör pip install -r requirements.txt
    3. Kör streamlit run KabbelAI.py