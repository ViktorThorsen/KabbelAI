## Käbbel-AI
En intelligent analytikerplattform för sökning, visualisering och analys av debatter och partiprogram från den svenska riksdagen (2012–2026). Systemet kombinerar modern sökteknik med stora språkmodeller (LLM) för att omvandla omfattande politisk data till strukturerade insikter.

---

### Systemarkitektur
Projektet är uppbyggt kring ett arbetsflöde i tre steg som utnyttjar två olika typer av AI-modeller.

#### Steg 1: Frågedekonstruktion (Router)
**Modell:** Gemini Flash 2.0
Det första steget fungerar som en router som analyserar användarens fråga för att bestämma:
* **Tidsperiod:** Om analysen ska fokusera på nuvarande mandatperiod eller historisk data.
* **Statistikbehov:** Om frågan kräver kvantitativ data (t.ex. "vem pratar mest om...").
* **Källval:** Om systemet ska prioritera officiella partiprogram (ideologi) eller debattprotokoll (faktiska handlingar).

#### Steg 2: Informationshämtning (Retrieval)
**Modell:** paraphrase-multilingual-MiniLM-L12-v2
Detta steg använder sökparametrarna från steg 1 för att identifiera och hämta relevanta dokument från en vektordatabas genom semantisk sökning.

#### Steg 3: Sammanställning och Svar (Synthesis)
**Modell:** Gemini Flash 2.0
En andra instans av Gemini Flash tar informationen från steg 2 och sammanställer den till ett strukturerat, faktabaserat svar som direkt adresserar användarens ursprungliga fråga.

---

### Installation och konfiguration

#### 1. Miljövariabler
Skapa en fil med namnet `.env` i projektets rotmapp. Lägg till din API-nyckel enligt följande format:
GEMINI_API_KEY=[DIN_GEMINI_API_NYCKEL]

#### 2. Skapa en virtuell miljö och installera nödvändiga bibliotek
python -m venv venv
pip install -r requirements.txt

#### 3. Starta applikationen
streamlit run KabbelAI.py
