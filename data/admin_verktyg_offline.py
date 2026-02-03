import chromadb
from chromadb.utils import embedding_functions
import os
import numpy as np

# --- INSTÄLLNINGAR ---
# Kontrollera att denna mapp-sökväg stämmer med din huvuddatabas
DB_PATH = "debatt_db" 
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def admin_panel():
    print("\n" + "="*60)
    print("🕵️  ADMIN-DETEKTOR: RÅDATA-ANALYS (OFFLINE)")
    print("="*60)
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Kunde inte hitta mappen '{DB_PATH}'.")
        return

    # Starta ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    
    try:
        collection = chroma_client.get_collection(name="riksdagen", embedding_function=local_ef)
    except Exception as e:
        print(f"❌ Fel vid laddning av collection: {e}")
        return

    while True:
        count = collection.count()
        print(f"\n📊 STATUS: {count} dokument i '{DB_PATH}'")
        print("-" * 30)
        print("1. 🔎 Sök & Jämför (Parti + Ämne + År)")
        print("2. 🔍 Inspektera specifikt ID")
        print("3. 🧹 Radera via Metadata (Typ/År/Parti)")
        print("4.  Ordsök")
        print("q. Avsluta")
        
        val = input("\nVälj alternativ: ")

        if val.lower() == "q": 
            break

        # --- 1. SÖK & JÄMFÖR (Den manuella hyckleri-detektorn) ---
        elif val == "1":
            parti = input("Ange parti (S, M, SD, C, V, KD, L, MP): ").upper()
            amne = input("Ange ämne/sökord: ")
            y_start = int(input("Startår (t.ex. 2012): "))
            y_end = int(input("Slutår (t.ex. 2024): "))

            print(f"\n📡 Söker efter '{amne}' för {parti}...")
            
            # Vi hämtar brett men filtrerar HÅRT på metadata-parti
            results = collection.query(
                query_texts=[amne],
                n_results=100,
                where={"parti": parti} # <--- Metadata-filtrering
            )

            early_docs = []
            late_docs = []

            if not results['documents'][0]:
                print("❌ Inga träffar för det partiet/ämnet.")
                continue

            for doc, meta, doc_id in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
                try:
                    year = int(str(meta.get('datum', meta.get('år', '0')))[:4])
                except: continue

                # Samla inlägg nära våra år (radie på 1 år)
                entry = f"ID: {doc_id} | Datum: {meta.get('datum')} | Talare: {meta.get('talare')}\n   \"{doc[:300]}...\""
                
                if abs(year - y_start) <= 1:
                    early_docs.append(entry)
                if abs(year - y_end) <= 1:
                    late_docs.append(entry)

            print(f"\n--- 🕰️ RESULTAT RUNT {y_start} ({len(early_docs)} st) ---")
            for d in early_docs[:3]: print(d + "\n")

            print(f"--- 🔥 RESULTAT RUNT {y_end} ({len(late_docs)} st) ---")
            for d in late_docs[:3]: print(d + "\n")

        # --- 2. INSPEKTERA ID ---
        elif val == "2":
            sok_id = input("Ange ID: ")
            res = collection.get(ids=[sok_id])
            if res['ids']:
                print(f"\n📄 ID: {res['ids'][0]}")
                print(f"Metadata: {res['metadatas'][0]}")
                print(f"Text: {res['documents'][0]}")
            else:
                print("❌ Hittades ej.")

        # --- 3. RADERA VIA METADATA ---
        elif val == "3":
            key = input("Radera via (typ/år/parti): ").lower()
            val_to_delete = input(f"Värde för {key}: ")
            
            # Konvertera år till sträng om det behövs (beror på hur du sparade det)
            confirm = input(f"⚠️ Är du helt säker på att radera ALLA {key}='{val_to_delete}'? (ja/nej): ")
            if confirm.lower() == "ja":
                collection.delete(where={key: val_to_delete})
                print("✅ Radering slutförd.")
        
        # --- SMART ORD-SÖKNING (Söker i ALLT för ett visst år) ---
        elif val == "4":
            ordet = input("Vilket ord letar du efter? (t.ex. 'invandring'): ").lower()
            ar = input("Vilket år? ")
            parti_filter = input("Filtrera på parti (S/M/SD/etc) eller tryck ENTER för alla: ").upper()
            
            print(f"⏳ Scannar databasen efter '{ordet}' år {ar}...")
            
            # Vi hämtar dokument och metadatas (men INTE 'ids' i include-listan)
            # Vi hämtar ALLA för det året först
            all_data = collection.get(
                where={"år": ar}, 
                include=['documents', 'metadatas'] # Tog bort 'ids' här!
            )
            
            if not all_data['documents']:
                print(f"❌ Hittade ingen data alls för år {ar}. Kontrollera att året är sparat som metadata.")
                continue

            hits = 0
            # ChromaDB skickar alltid med 'ids' i en egen lista i all_data
            for d, m, i in zip(all_data['documents'], all_data['metadatas'], all_data['ids']):
                # Kolla om ordet finns i texten ELLER i rubriken (som ofta ligger i metadata)
                texten_matchar = ordet in d.lower()
                metadata_matchar = ordet in str(m).lower()
                
                # Kolla partifilter
                parti_matchar = True
                if parti_filter and m.get('parti') != parti_filter:
                    parti_matchar = False

                if (texten_matchar or metadata_matchar) and parti_matchar:
                    print(f"\n🎯 TRÄFF! ID: {i}")
                    print(f"Talare: {m.get('talare')} ({m.get('parti')}) | Datum: {m.get('datum')}")
                    print(f"Text: {d[:200]}...")
                    hits += 1
                    if hits >= 20: 
                        print("\n...visar de 20 första träffarna. Det finns troligen fler.")
                        break
            
            if hits == 0:
                print(f"❌ Inga träffar för '{ordet}' hos {parti_filter if parti_filter else 'något parti'} under {ar}.")

if __name__ == "__main__":
    admin_panel()