import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
import os
import glob
import re

# --- INSTÄLLNINGAR ---
DB_PATH = "data/debatt_db" 
PDF_MAPP = "data/partiprogram"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def load_program():
    if not os.path.exists(PDF_MAPP):
        print(f"❌ Mappen '{PDF_MAPP}' saknas!")
        return

    # Starta DB
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    
    collection = client.get_or_create_collection(name="riksdagen", embedding_function=ef)

    pdf_filer = glob.glob(os.path.join(PDF_MAPP, "*.pdf"))
    if not pdf_filer:
        print("Inga PDF-filer hittades.")
        return

    all_ids = []
    all_docs = []
    all_metas = []

    for pdf_path in pdf_filer:
        filnamn = os.path.basename(pdf_path)
        
        parti_match = re.search(r'^([A-ZÅÄÖ]+)', filnamn.upper())
        ar_match = re.search(r'(\d{4})', filnamn)
        
        parti = parti_match.group(1) if parti_match else "OKÄNT"
        ar = ar_match.group(1) if ar_match else "2024"
        
        print(f"Läser {parti} ({ar})...")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n\n"
                
                raw_chunks = full_text.split("\n\n")
                clean_chunks = []
                current_chunk = ""
                
                for bit in raw_chunks:
                    bit = bit.strip().replace("\x00", "")
                    if len(bit) < 40: continue 
                    
                    if len(current_chunk) + len(bit) < 1200:
                        current_chunk += " " + bit
                    else:
                        clean_chunks.append(current_chunk.strip())
                        current_chunk = bit
                
                if current_chunk:
                    clean_chunks.append(current_chunk.strip())

                for i, chunk in enumerate(clean_chunks):
                    doc_id = f"prog_{parti}_{ar}_{i}"
                    
                    enhanced_text = f"PARTIPROGRAM ({parti}, {ar}): {chunk}"
                    meta = {
                        "typ": "program", 
                        "parti": parti,
                        "år": ar,
                        "källa": filnamn,
                        "dok_id": f"PROG-{parti}-{ar}",
                        "nummer": i
                    }
                    
                    all_ids.append(doc_id)
                    all_docs.append(enhanced_text)
                    all_metas.append(meta)

        except Exception as e:
            print(f"Fel vid läsning av {filnamn}: {e}")

    # Spara till ChromaDB
    if all_ids:
        print(f"\nSparar {len(all_ids)} program-chunks...")
        batch_size = 100
        for i in range(0, len(all_ids), batch_size):
            end = min(i + batch_size, len(all_ids))
            collection.upsert(
                ids=all_ids[i:end],
                documents=all_docs[i:end],
                metadatas=all_metas[i:end]
            )
        print(f"Klart! Totalt antal dokument i databasen nu: {collection.count()}")
    else:
        print("Ingen ny data hittades.")

if __name__ == "__main__":
    load_program()