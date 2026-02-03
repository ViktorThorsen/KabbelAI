import chromadb
from chromadb.utils import embedding_functions
import json
import os
import hashlib

DB_PATH = "data/debatt_db" 
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

FILER_ATT_LADDA = [
    "data/anforanden/riksdags_debatter.jsonl"
]

EXTRA_SOKORD_REGLER = [
    {
        "beskrivning": "Tidölaget (Regeringen + SD efter 2022)",
        "start_datum": "2022-10-18",
        "slut_datum": "2030-01-01",
        "partier": ["M", "KD", "L", "SD"],
        "text": " [Tidölaget, Regeringsunderlaget, Tidöavtalet]"
    },
    {
        "beskrivning": "Oppositionen (Efter 2022)",
        "start_datum": "2022-10-18",
        "slut_datum": "2030-01-01",
        "partier": ["S", "V", "MP", "C"],
        "text": " [Oppositionen, De rödgröna]"
    },
    {
        "beskrivning": "Januariavtalet (S+MP+C+L under förra mandaten)",
        "start_datum": "2019-01-01",
        "slut_datum": "2021-06-21",
        "partier": ["S", "MP", "C", "L"],
        "text": " [Januariavtalet, JÖK]"
    },
    {
        "beskrivning": "Alliansen (Historiskt)",
        "start_datum": "2006-10-06",
        "slut_datum": "2014-10-03",
        "partier": ["M", "C", "L", "KD"],
        "text": " [Alliansen, Borgerliga regeringen]"
    }
]

def hämta_extra_sökord(datum, parti):
    if not datum or not parti: return ""
    extra_text = ""
    for regel in EXTRA_SOKORD_REGLER:
        start = regel.get("start_datum", "0000-00-00")
        slut = regel.get("slut_datum", "9999-99-99")
        if start <= datum <= slut and parti in regel["partier"]:
            extra_text += regel["text"]
    return extra_text

def processa_rad(data):
    if "talare" in data and "dok_id" in data:
        typ = "debatt"
        doc_id = data.get("id", f"{data['dok_id']}-{data.get('nummer', '0')}")
        
        datum = data.get("datum", "0000-00-00")
        parti = data.get("parti", "")
        rubrik = data.get("rubrik", "")
        
        extra_ord = hämta_extra_sökord(datum, parti)
        
        text_content = f"RUBRIK: {rubrik}\nTALARE: {data['talare']} ({parti})\nTEXT: {data['text']}{extra_ord}"
        
        meta = { 
            "typ": typ, 
            "talare": data['talare'], 
            "parti": parti, 
            "datum": datum, 
            "år": datum.split("-")[0], 
            "dok_id": data['dok_id'],
            "nummer": data.get("nummer", 0),
            "rubrik": rubrik, 
            "replik": data.get("ar_replik", "N") 
        }
        return doc_id, text_content, meta
    return None

def ladda_databas():
    print(f"🔨 Skapar renodlad DEBATT-databas i: {DB_PATH}")
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

    collection = chroma_client.get_or_create_collection(
        name="riksdagen",
        embedding_function=local_ef
    )

    all_ids = []
    all_docs = []
    all_metas = []

    print("📖 Läser in debattfiler...")
    for filnamn in FILER_ATT_LADDA:
        if not os.path.exists(filnamn): 
            print(f" ⚠️ Varning: {filnamn} saknas.")
            continue
            
        print(f"   -> Bearbetar {filnamn}...")
        
        count = 0
        with open(filnamn, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    res = processa_rad(json.loads(line))
                    if res:
                        doc_id, text, meta = res
                        all_ids.append(doc_id)
                        all_docs.append(text)
                        all_metas.append(meta)
                        count += 1
                except: continue
        print(f"   -> Hittade {count} anföranden.")

    if not all_ids:
        print("❌ Ingen data hittades att spara.")
        return

    batch_size = 200
    total = len(all_ids)
    print(f"\n💾 Vektoriserar och sparar {total} dokument...")
    print("(Detta kan ta en stund eftersom AI:n måste läsa varje text...)")

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        print(f"   Processing {i} - {end} ...", end="\r")
        
        collection.upsert(
            ids=all_ids[i:end],
            documents=all_docs[i:end],
            metadatas=all_metas[i:end]
        )
        
    print(f"\n✅ KLART! Din nya debatt-hjärna ligger i '{DB_PATH}'")

if __name__ == "__main__":
    ladda_databas()