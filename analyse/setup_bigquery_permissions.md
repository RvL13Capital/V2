# BigQuery Berechtigungen einrichten - Schritt für Schritt

## Option 1: Berechtigungen zum bestehenden Service Account hinzufügen

1. **Google Cloud Console öffnen:**
   - https://console.cloud.google.com
   - Projekt: `ignition-ki-csv-storage` auswählen

2. **IAM & Admin → Service Accounts:**
   - Finden Sie Ihren Service Account (der mit `-e7bb9d0fd1d0.json` endet)
   - Klicken Sie auf den Service Account

3. **Berechtigungen hinzufügen:**
   - Klicken Sie auf "PERMISSIONS" Tab
   - Klicken Sie auf "GRANT ACCESS"
   - Fügen Sie diese Rollen hinzu:
     * `BigQuery Data Editor` - Für Daten lesen/schreiben
     * `BigQuery Job User` - Für Query-Ausführung
     * `BigQuery User` - Für Dataset-Erstellung

4. **Speichern und warten:**
   - Klicken Sie auf "SAVE"
   - Warten Sie 1-2 Minuten bis die Berechtigungen aktiv sind

## Option 2: BigQuery API aktivieren (falls noch nicht aktiv)

1. **APIs & Services → Enable APIs:**
   - Suchen Sie nach "BigQuery API"
   - Klicken Sie auf "ENABLE"

2. **Kosten-Kontrolle einrichten (Optional aber empfohlen):**
   - Billing → Budgets & alerts
   - Create budget: $1 pro Monat
   - Alert bei 50%, 90%, 100%

## Option 3: Neues Service Account nur für BigQuery

```bash
# In Google Cloud Shell ausführen:

# Service Account erstellen
gcloud iam service-accounts create bigquery-analyzer \
    --display-name="BigQuery Market Analyzer"

# Email des Service Accounts
export SA_EMAIL=bigquery-analyzer@ignition-ki-csv-storage.iam.gserviceaccount.com

# Berechtigungen zuweisen
gcloud projects add-iam-policy-binding ignition-ki-csv-storage \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding ignition-ki-csv-storage \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding ignition-ki-csv-storage \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectViewer"

# Key erstellen und herunterladen
gcloud iam service-accounts keys create ~/bigquery-key.json \
    --iam-account=${SA_EMAIL}
```

## Kostenlose Nutzung sicherstellen:

### Quotas setzen (Empfohlen):
1. **BigQuery → Admin → Quotas:**
   - Maximum bytes billed per day: 33GB (1TB/30 Tage)
   - Maximum bytes billed per query: 10GB

### Monitoring einrichten:
1. **Billing → Budget alerts:**
   - Budget: $0 (nur Free Tier)
   - Alert bei jeder Nutzung über Free Tier

## Kosten-Beispiele für Ihre Nutzung:

| Anzahl Aktien | Daten-Größe | Kosten |
|---------------|-------------|--------|
| 100 | ~3 GB | $0 (Free Tier) |
| 500 | ~15 GB | $0 (Free Tier) |
| 1000 | ~30 GB | $0 (Free Tier) |
| 5000 | ~150 GB | $0 (Free Tier) |
| 35000 | ~1 TB | $0 (Grenze Free Tier) |
| 50000 | ~1.5 TB | $2.50 |

## Tägliche Nutzung mit Free Tier:

- **33 GB pro Tag** = ~1000 Aktien täglich analysieren
- **1 TB pro Monat** = ~35000 Aktien monatlich
- Alles **kostenlos** im Free Tier!

## Wichtige Hinweise:

1. **Storage in BigQuery:** Die ersten 10GB sind kostenlos
2. **Streaming:** 1TB/Monat kostenlos
3. **Queries:** Unbegrenzt viele, aber max 1TB Daten/Monat kostenlos
4. **Keine Kreditkarte nötig** für Free Tier wenn Projekt schon existiert

## Befehle zum Testen:

```python
# Test ob BigQuery funktioniert:
from google.cloud import bigquery

client = bigquery.Client(project="ignition-ki-csv-storage")
datasets = list(client.list_datasets())
print(f"Datasets: {len(datasets)}")

# Test Query (kostet nichts):
query = "SELECT 1 as test"
result = client.query(query)
print("BigQuery works!")
```