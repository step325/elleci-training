# 🧪 Data Quality Pipeline (Sperimentale)

Questa cartella contiene gli strumenti per generare e filtrare dati di alta qualità ("Textbook Quality") per Elleci.

### 1. Requirements
```bash
pip install google-genai tqdm
```(Opzione A)
L'obiettivo è generare dati che sembrano "libri di testo": chiari, didattici, strutturati.

### Workflow Proposto
1.  **Topic/Seed**: Creare una lista di argomenti (es. da Wikipedia o Syllabus scolastici).
2.  **Generator**: Usare un LLM esterno (GPT-4o, Claude 3.5 Sonnet, Llama-3-70B) via API.
3.  **Prompt**:
    ```text
    Sei un professore universitario esperto e un divulgatore eccellente.
    Scrivi una spiegazione dettagliata, chiara e didattica sull'argomento: "{topic}".
    Usa markdown. Includi esempi, definizioni e, se applicabile, codice.
    Stile: Manuale scolastico moderno. Lingua: Italiano.
    ```
4.  **Processing**: Salvare in `.jsonl`.

### Esempio
Vedi `data/synthetic_textbooks_it_sample.jsonl` per l'output desiderato.

## 2. Quality Filtering (Opzione B)
Filtrare dati massivi (es. CulturaX) usando metriche di qualità.

### Metodi
-   **Perplexity Filter**: Scartare testi dove un modello piccolo ha perplessità troppo alta (rumore) o troppo bassa (ripetizioni).
-   **Instruct-Score**: Usare un modello come `Qwen-2.5-7B` per dare un voto 1-5 al valore educativo.

## 3. Usage
Per caricare i dati sintetici durante il training, aggiungere `SyntheticDataset` al mix in `scripts/train_elleci.py`.
