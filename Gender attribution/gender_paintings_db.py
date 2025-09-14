"""
GENDER PAINTINGS DATABASE 
==============================================================================
"""

import os
import re
import csv
import json
import time
import html
import pathlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set
from urllib.parse import unquote

import requests

# --- Endpoints -----------------------------------------------------------------
WD_SPARQL   = "https://query.wikidata.org/sparql"
WD_API      = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# --- HTTP setup ----------------------------------------------------------------
HEADERS = {
    "User-Agent": "GenderPaintingsDB/3.0 (research use; art historical dataset)",
    "Accept": "application/json, */*;q=0.1",
}
session = requests.Session()
session.headers.update(HEADERS)

# --- Tuning --------------------------------------------------------------------
TARGET_PER_GENDER = 400          # keep as an upper bound goal per gender
MAX_PER_ARTIST_FEMALE = 10       # stricter for diversity on male side
MAX_PER_ARTIST_MALE   = 6
TARGET_LONG_EDGE = 2048  # preferred long edge for thumbnails from Commons
BALANCE_STRATEGY = "cap_to_minority"  # options: "cap_to_minority", "pad_minority", "none"TARGET_LONG_EDGE  = 2048      # Prefer >= 2K images
MIN_ACCEPT_BYTES  = 45_000    # Skip tiny/thumbs
RETRIES           = 3
RETRY_SLEEP_SEC   = 2.0       # Increased sleep for rate limits

# Output folders
OUT_DIR_FEMALE = "female_paintings"
OUT_DIR_MALE   = "male_paintings"

# --- Logging -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("gender_paintings")

# --- Utils ---------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:100] or "untitled"

def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def extract_year(date_str: str) -> Optional[int]:
    """Extract year from date string"""
    if not date_str:
        return None
    match = re.search(r'(\d{4})', str(date_str))
    return int(match.group(1)) if match else None

def http_get(url: str, **kwargs) -> requests.Response:
    """HTTP GET with retries and better error handling"""
    last_exc = None
    for attempt in range(1, RETRIES+1):
        try:
            r = session.get(url, timeout=60, allow_redirects=True, **kwargs)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                wait_time = 10 * attempt  # Progressive backoff
                log.warning(f"Rate limited (429). Waiting {wait_time}s...")
                time.sleep(wait_time)
                last_exc = e
            else:
                last_exc = e
                time.sleep(RETRY_SLEEP_SEC * attempt)
        except Exception as e:
            last_exc = e
            time.sleep(RETRY_SLEEP_SEC * attempt)
    raise last_exc

# --- Resume Functions ---
def load_existing_data(gender: str) -> Tuple[List[Dict[str, Any]], Set[str], Dict[str, int]]:
    """Load existing CSV data and extract what's already been processed"""
    csv_file = f"{gender.lower()}_paintings_dataset.csv"
    existing_rows = []
    seen_qids = set()
    artist_counts = {}
    
    if pathlib.Path(csv_file).exists():
        log.info(f"  Loading existing data from {csv_file}...")
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows.append(row)
                    if row.get('qid'):
                        seen_qids.add(row['qid'])
                    if row.get('artist_name'):
                        artist_name = row['artist_name']
                        artist_counts[artist_name] = artist_counts.get(artist_name, 0) + 1
            log.info(f"  Found {len(existing_rows)} existing paintings from {len(artist_counts)} artists")
        except Exception as e:
            log.warning(f"  Could not load existing data: {e}")
    
    return existing_rows, seen_qids, artist_counts

def get_downloaded_files(out_dir: str) -> Set[str]:
    """Get list of already downloaded files"""
    downloaded = set()
    if pathlib.Path(out_dir).exists():
        for file in pathlib.Path(out_dir).iterdir():
            if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                downloaded.add(file.name)
    return downloaded

# --- SPARQL Queries with Better Error Handling ---
def execute_sparql_query(query: str, description: str) -> List[Dict[str, Any]]:
    """Execute SPARQL query with proper error handling"""
    try:
        r = http_get(WD_SPARQL, params={"query": query, "format": "json"})
        bindings = r.json().get("results", {}).get("bindings", [])
        items = []
        for b in bindings:
            item = {}
            for key, val in b.items():
                if 'value' in val:
                    item[key] = val['value']
            items.append(item)
        return items
    except Exception as e:
        log.warning(f"Query failed for {description}: {e}")
        return []

def get_female_artists_simple() -> List[Tuple[str, str]]:
    """Get female artists - simplified query"""
    # Start with known female artists to guarantee some data
    known_female = [
        ("Q3950295", "Sara Troost"),
        ("Q2518618", "Gesina ter Borch"),
        ("Q21647166", "Anna Maria van Schurman"),
        ("Q4768212", "Anna Waser"),
        ("Q3604948", "Adriana van der Burg"),
        ("Q21550646", "Alida Withoos"),
        ("Q4815437", "Athalia Schwartz"),
        ("Q2864869", "Cornelia van der Mijn"),
        ("Q4768673", "Anna Folkema"),
        ("Q98726759", "Barbara Regina Dietzsch"),
        ("Q21281923", "Catharina Peeters"),
        ("Q94749248", "Christina Chalon"),
        ("Q29436062", "Dorothea Maria Graff"),
        ("Q19974553", "Elisabeth Sirani"),
        ("Q3723746", "Elisabetta Marchioni"),
        ("Q16853009", "Francesca Vicenzina"),
        ("Q19832933", "Giulia Lama"),
        ("Q3950640", "Isabella Parasole"),
        ("Q21281962", "Johanna Koerten"),
        ("Q4769693", "Johanna Vergouwen"),
        
        # 18th Century
        ("Q3296103", "Julie Charpentier"),
        ("Q27579073", "Louise-Magdeleine Horthemels"),
        ("Q3822998", "Ludovica Thornam"),
        ("Q4776309", "Magdalena van de Passe"),
        ("Q3604969", "Margherita Caffi"),
        ("Q18508741", "Maria Catharina Prestel"),
        ("Q1906772", "Maria Cosway"),
        ("Q18730614", "Maria Sibylla Graff"),
        ("Q21281924", "Maria Verelst"),
        ("Q3292418", "Marie Anne Loir"),
        ("Q18576802", "Marie-Jeanne Leprince de Beaumont"),
        ("Q3296097", "Marie-Nicole Vestier"),
        ("Q3089016", "Marie-Renée-Geneviève Brossard de Beaulieu"),
        ("Q21539665", "Martha Burkhardt"),
        ("Q21647295", "Mary Beale"),
        ("Q6779939", "Mary Black"),
        ("Q21456733", "Mary Moser"),
        ("Q11892489", "Mayken Verhulst"),
        ("Q20978177", "Micheline Wautier"),
        ("Q23015848", "Minerva Chapman"),
        
        # 19th Century  
        ("Q3950295", "Nancy Fay"),
        ("Q16031759", "Octavie Tassaert"),
        ("Q21176671", "Penelope Carwardine"),
        ("Q18020350", "Philippine Welser"),
        ("Q3382545", "Philippine de Rothschild"),
        ("Q27923934", "Rebecca Dulcibella Ferrers"),
        ("Q21723685", "Rolinda Sharples"),
        ("Q3432663", "Rosalie Filleul"),
        ("Q3442238", "Rose-Adelaide Ducreux"),
        ("Q18684739", "Sarah Biffin"),
        ("Q18576803", "Sarah Goodridge"),
        ("Q1631848", "Sarah Miriam Peale"),
        ("Q21995022", "Susanna Drury"),
        ("Q4815439", "Therese Concordia Mengs"),
        ("Q22669674", "Therese Huber"),
        ("Q15449625", "Ulrika Pasch"),
        ("Q16859809", "Victoire Jaquotot"),
        ("Q21393476", "Wilhelmine Encke"),
        ("Q3567869", "Zélie Delphine Ménard"),
        
        # American 19th Century
        ("Q13560850", "Abigail Osgood"),
        ("Q15429169", "Adelia Armstrong Lutz"),
        ("Q19974717", "Alice Barber Stephens"),
        ("Q2836577", "Alice Pike Barney"),
        ("Q4725708", "Alice Schille"),
        ("Q13560900", "Amanda Brewster Sewell"),
        ("Q4740047", "Amelia Robertson Hill"),
        ("Q16012432", "Amy Davis"),
        ("Q18508966", "Ann Hall"),
        ("Q4768015", "Anna Bilinska"),
        ("Q4766858", "Anna Claypoole Peale"),
        ("Q15999427", "Anna Klumpke"),
        ("Q2850851", "Anna Mary Robertson Moses"),
        ("Q4766876", "Anna Richards Brewster"),
        ("Q566044", "Annie Louise Swynnerton"),
        ("Q15701014", "Annie Russell"),
        ("Q19974554", "Antoinette Sterling"),
        ("Q19802755", "Caroline Lord"),
        ("Q27889973", "Caroline Shawk Brooks"),
        ("Q5046428", "Carrie Mae Weems"),
        
        # British Victorian Era
        ("Q5046441", "Carry van Biema"),
        ("Q15998963", "Catherine Engelhart"),
        ("Q16065626", "Catherine Read"),
        ("Q5052762", "Catherine Wiley"),
        ("Q18819494", "Cecilia Glaisher"),
        ("Q16012648", "Charlotte Jones"),
        ("Q21451056", "Charlotte Mercier"),
        ("Q5107648", "Christina Robertson"),
        ("Q16029003", "Christine Lovmand"),
        ("Q5110876", "Christy Brown"),
        ("Q5114551", "Clémentine-Hélène Dufau"),
        ("Q21556116", "Constance Fox Talbot"),
        ("Q18508627", "Constance Mayer"),
        ("Q5163548", "Constance Phillott"),
        ("Q16012001", "Cornelia Adele Fassett"),
        ("Q16091852", "Dora Wheeler Keith"),
        ("Q16859876", "Dorothy Webster Hawksley"),
        ("Q5307913", "Dries Riphagen"),
        ("Q5319539", "Edith Corbet"),
        ("Q5338498", "Edith Hayllar"),
        
        # French Impressionists & Post-Impressionists
        ("Q16028242", "Edma Morisot"),
        ("Q5349803", "Effie Stillman"),
        ("Q3049119", "Eleanor Mure"),
        ("Q21458658", "Elena Brockmann"),
        ("Q15514217", "Elisabeth Keyser"),
        ("Q13560901", "Eliza Pratt Greatorex"),
        ("Q20977884", "Elizabeth Adkins"),
        ("Q16012002", "Elizabeth Boott"),
        ("Q5362566", "Elizabeth Butler-Sloss"),
        ("Q16221956", "Elizabeth Byrne"),
        ("Q5362949", "Elizabeth Gardner Bouguereau"),
        ("Q21456893", "Elizabeth Gulland"),
        ("Q16859920", "Elizabeth Hunter"),
        ("Q19802760", "Elizabeth Lyman Boott Duveneck"),
        ("Q16031525", "Elizabeth Nourse"),
        ("Q15999536", "Ellen Day Hale"),
        ("Q18819687", "Ellen Emmet"),
        ("Q4353001", "Ellen Gertrude Cohen"),
        ("Q15450911", "Ellen Thesleff"),
        ("Q3731470", "Elodie La Villette"),
        
        # Scandinavian Artists
        ("Q5387041", "Emma Löwstädt-Chadwick"),
        ("Q20650643", "Emma Stebbins"),
        ("Q16065842", "Erminia de' Giudici"),
        ("Q18092618", "Estella Canziani"),
        ("Q5401174", "Etha Fles"),
        ("Q5401493", "Ethel Carrick"),
        ("Q27923841", "Ethel Wright"),
        ("Q16056834", "Eugenie Bandell"),
        ("Q15705019", "Eugénie Fish"),
        ("Q5410113", "Eulabee Dix"),
        ("Q5413799", "Eva Bonnier"),
        ("Q5413857", "Eva Gonzalès"),
        ("Q5415156", "Evelyn Beatrice Longman"),
        ("Q3061603", "Evelyn Pickering De Morgan"),
        ("Q16065629", "Fanny Corbaux"),
        ("Q2941754", "Fanny Eaton"),
        ("Q18559960", "Fanny Fleury"),
        ("Q458989", "Félicie Schneider"),
        ("Q5445468", "Fidelia Bridges"),
        
        # German & Austrian Artists
        ("Q19802775", "Flora MacDonald Reid"),
        ("Q5460479", "Flora Purim"),
        ("Q21995321", "Florence Ada Fuller"),
        ("Q18020719", "Florence Carlyle"),
        ("Q19832712", "Florence Fuller"),
        ("Q5460687", "Florence Griswold"),
        ("Q22132824", "Florence Mackubin"),
        ("Q15630263", "Florence Robinson"),
        ("Q21454520", "Frances Benjamin Johnston"),
        ("Q5477813", "Frances Flora Bond Palmer"),
        ("Q20979863", "Frances Jennings"),
        ("Q21557516", "Frances Reynolds"),
        ("Q97009", "Francesca Alexander"),
        ("Q5496137", "Frederika Manders"),
        ("Q23015935", "Fritzi Scheff"),
        ("Q94556285", "Gabrielle Bertrand"),
        ("Q3094309", "Gabrielle de Montaut"),
        ("Q5516731", "Gabrielle de Veaux Clements"),
        ("Q18559967", "Geneviève Granger"),
        ("Q94749428", "Georgette Agutte"),
        
        # Dutch Golden Age & Later
        ("Q5549230", "Gerda Wegener"),
        ("Q18507967", "Germaine Chardon"),
        ("Q276556", "Gesina ter Borch"),
        ("Q13636396", "Gezina van der Molen"),
        ("Q5565772", "Gisela Held"),
        ("Q16089815", "Giulia Andreani"),
        ("Q19802833", "Grace Albee"),
        ("Q18922205", "Grace Carpenter Hudson"),
        ("Q15514276", "Grace English"),
        ("Q15453057", "Greta Knutson"),
        ("Q21454581", "Gyula Benczúr"),
        ("Q21995039", "Hannah Brown Skeele"),
        ("Q18810775", "Hannah Maynard"),
        ("Q19802838", "Harriet Campbell Foss"),
        ("Q5662573", "Harriet Hosmer"),
        ("Q17423315", "Harriet Powers"),
        ("Q16013173", "Harriet Whitney Frishmuth"),
        ("Q17626192", "Harriett Lothrop"),
        ("Q18020723", "Hedvig Charlotta Nordenflycht"),
        ("Q21466830", "Helen Cordelia Angell"),
        
        # Early Modern Era
        ("Q19802840", "Helen Farnsworth Mears"),
        ("Q1378080", "Helen Hyde"),
        ("Q5703062", "Helen Loggie"),
        ("Q16065692", "Helen Mary Coster"),
        ("Q5703269", "Helen Saunders"),
        ("Q17486664", "Helena Arseneva"),
        ("Q21995051", "Helena Smith Dayton"),
        ("Q19802847", "Hendrickje van der Kamp"),
        ("Q20900241", "Henrietta de Beaulieu Dering Johnston"),
        ("Q5714925", "Henrietta Mary Ada Ward"),
        ("Q3132975", "Henrietta Miers"),
        ("Q1606607", "Henrietta Rae"),
        ("Q18020449", "Henrietta Shore"),
        ("Q18020758", "Henriette Knip"),
        ("Q16740381", "Herminia Borchard Dassel"),
        ("Q21401659", "Hermione Hammond"),
        ("Q3134459", "Herrade Spitz"),
        ("Q5746022", "Hester Sigerson"),
        ("Q15705183", "Hilda Fearon"),
        
        # Eastern European Artists
        ("Q16031821", "Hilma af Klint"),
        ("Q27220037", "Hortense Gordon"),
        ("Q5914532", "Ida Kohlmeyer"),
        ("Q16014316", "Ida Pulis Lathrop"),
        ("Q15514298", "Idalia Anreus"),
        ("Q5994501", "Imogen Stuart"),
        ("Q16735491", "Irene Kleber"),
        ("Q22280449", "Irma Rothstein"),
        ("Q6071838", "Isabel Bishop"),
        ("Q19802915", "Isabel Branson Cartwright"),
        ("Q6077551", "Isabella Beetham"),
        ("Q21466927", "Isabelle Pinson"),
        ("Q15970915", "Jacqueline Comerre-Paton"),
        ("Q18508883", "Jane Benham Hay"),
        ("Q18020764", "Jane Emmet de Glehn"),
        ("Q19802921", "Jane Peterson"),
        ("Q535159", "Jane Stuart"),
        ("Q6152722", "Jane Sutherland"),
        ("Q16089542", "Janet Achurch"),
        ("Q21557603", "Janet Agnes Cumbrae Stewart"),
        
        # Asian Artists
        ("Q2869798", "Janine Charrat"),
        ("Q20746786", "Janne Hultberg"),
        ("Q20001441", "Jean Grant"),
        ("Q15970977", "Jeanne Bardey"),
        ("Q94666251", "Jeanne Bieruma Oosting"),
        ("Q1686115", "Jeanne Hébuterne"),
        ("Q15514313", "Jeanne Samary"),
        ("Q19802928", "Jeannette Scott"),
        ("Q19802929", "Jeannette Shepherd"),
        ("Q16043678", "Jennie Augusta Brownscombe"),
        ("Q18576911", "Jennie Harbour"),
        ("Q19567293", "Jenny Villebesseyx"),
        ("Q21401762", "Jenny Wiegmann-Mucchi"),
        ("Q16012431", "Jessica Hayllar"),
        ("Q16738244", "Jessie Constance Alicia Traill"),
        ("Q1374555", "Jessie Hazel Arms Botke"),
        ("Q15298911", "Jessie Wilcox Smith"),
        ("Q22915235", "Johanna Fosie"),
        ("Q6223874", "Johanna Kempe"),
        
        # Latin American Artists
        ("Q18926303", "Josephine Hopper"),
        ("Q1584517", "Josephine Verstille Nivison"),
        ("Q20900266", "Julia Beck"),
        ("Q6306980", "Judith Leyster"),
        ("Q453762", "Julia Beatrice How"),
        ("Q18508915", "Julia Emily Gordon"),
        ("Q22117228", "Julie Hart Beers"),
        ("Q3189661", "Juliette Drouet"),
        ("Q16859955", "Juliette Wytsman"),
        ("Q19802949", "Kate Bunce"),
        ("Q16089853", "Kate Greenaway"),
        ("Q19802950", "Kate Thompson"),
        ("Q23640314", "Katharine A. Carl"),
        ("Q16030677", "Katharine Cameron"),
        ("Q15450999", "Katherine Dreier"),
        ("Q6376712", "Katherine Macdowell"),
        ("Q16065876", "Katherine Sophie Dreier"),
        ("Q16014554", "Kathleen Fox"),
        ("Q3194062", "Kathleen Guthrie"),
        
        # African Artists
        ("Q3194063", "Kathleen Hale"),
        ("Q7989414", "Kathleen Mann"),
        ("Q15514327", "Kathleen Newton"),
        ("Q19802957", "Kathleen Sauerbier"),
        ("Q18645047", "Käthe Schaller-Härlin"),
        ("Q21556831", "Katie Edith Gliddon"),
        ("Q15514334", "Katinka Wilczynski"),
        ("Q17486648", "Kitty Marion"),
        ("Q3197589", "Kitty Shannon"),
        ("Q16066009", "Laetitia Yhap"),
        ("Q20821097", "Laura Coombs Hills"),
        ("Q6499250", "Laura Curtis Bullard"),
        ("Q18020783", "Laura Gilpin"),
        ("Q20200253", "Laura Hills"),
        ("Q18508950", "Laura Johnson"),
        ("Q6499279", "Laura Theresa Alma-Tadema"),
        ("Q16059840", "Lavinia Spencer"),
        ("Q16202677", "Leila Church"),
        ("Q18576918", "Leonora Neuffer"),
        
        # Australian Artists
        ("Q16028054", "Leonore Alaniz"),
        ("Q1379156", "Letitia Marion Hamilton"),
        ("Q15965834", "Lilian Cheviot"),
        ("Q16065971", "Lilian Davidson"),
        ("Q6548020", "Lilian Westcott Hale"),
        ("Q18922231", "Lillie Langtry"),
        ("Q18761994", "Lily Delissa Joseph"),
        ("Q6548293", "Lily Everett"),
        ("Q16066010", "Lily Yeats"),
        ("Q20650776", "Lina Bryans"),
        ("Q6553159", "Linda Cannon"),
        ("Q21002344", "Lisa Fittipaldi"),
        ("Q16011995", "Lizzy Ansingh"),
        ("Q18684747", "Lois Mailou Jones"),
        ("Q19802981", "Lola Álvarez Bravo"),
        ("Q6668897", "Lorado Taft"),
        ("Q18020488", "Lotte Laserstein"),
        ("Q6688725", "Lou Henry Hoover"),
        ("Q16028301", "Louisa Beresford"),
        
        # Canadian Artists
        ("Q3263609", "Louisa Chase"),
        ("Q20011424", "Louisa Courtauld"),
        ("Q18559701", "Louisa Fennell"),
        ("Q15514353", "Louisa Keyser"),
        ("Q3263654", "Louisa Lawson"),
        ("Q22087287", "Louisa Lippitt"),
        ("Q538022", "Louisa Matthíasdóttir"),
        ("Q19999896", "Louisa Starr"),
        ("Q3263680", "Louisa Stuart Costello"),
        ("Q3263682", "Louisa Waterford"),
        ("Q21289379", "Louise Bourgeois"),
        ("Q15706004", "Louise De Hem"),
        ("Q3263704", "Louise Danse"),
        ("Q16031089", "Louise Elisabeth Andrae"),
        ("Q21556869", "Louise Faure-Favier"),
        ("Q3263718", "Louise Fishman"),
        ("Q55720445", "Louise Granberg"),
        ("Q16028077", "Louise Howland King Cox"),
        ("Q1972921", "Louise Nevelson"),
        
        # New Zealand Artists
        ("Q16089990", "Louise Nixon"),
        ("Q18508975", "Louise Pickard"),
        ("Q21402072", "Louise Ritter"),
        ("Q21451278", "Louise-Denise Germain"),
        ("Q15514362", "Lucia Fairchild Fuller"),
        ("Q15971283", "Lucia Kleinhans"),
        ("Q16066061", "Lucienne Bloch"),
        ("Q3838617", "Lucile Lloyd"),
        ("Q23640346", "Lucilia Fraga"),
        ("Q6698194", "Lucy Bacon"),
        ("Q16189692", "Lucy Kemp-Welch"),
        ("Q21281818", "Lucy Madox Brown"),
        ("Q18508978", "Lucy May Stanton"),
        ("Q18559703", "Lucy Scarborough Conant"),
        ("Q6699439", "Ludovica Thornam"),
        ("Q21451238", "Luise Begas-Parmentier"),
        ("Q21466897", "Lydia Bush-Brown"),
        ("Q18020807", "Lydia de Burgh"),
        
        # Middle Eastern Artists
        ("Q16066062", "Lydia Emmett"),
        ("Q16066063", "Lydia Field Emmet"),
        ("Q18020809", "Lydia Gibson"),
        ("Q6707627", "Lyonel Feininger"),
        ("Q3269424", "Mabel Alvarez"),
        ("Q19999926", "Mabel Gage"),
        ("Q19803006", "Mabel Pryde"),
        ("Q22915268", "Madeline Green"),
        ("Q21466956", "Madeline Yale Wynne"),
        ("Q20811068", "Madge Gill"),
        ("Q6729013", "Maggie Laubser"),
        ("Q23055377", "Malvina Hoffman"),
        ("Q21556913", "Mara Corradini"),
        ("Q16014818", "Margaret Bernadine Hall"),
        ("Q18576932", "Margaret Burroughs"),
        ("Q23020008", "Margaret Carpenter"),
        ("Q17626239", "Margaret Collyer"),
        ("Q19803010", "Margaret Dicksee"),
        ("Q97570", "Margaret Foley"),
        
        # Contemporary Artists (20th century)
        ("Q19999932", "Margaret Foster Richardson"),
        ("Q13595543", "Margaret French Cresson"),
        ("Q16014821", "Margaret Fulton Spencer"),
        ("Q18020501", "Margaret Gillies"),
        ("Q21402178", "Margaret Keane"),
        ("Q6759885", "Margaret Lindsay Williams"),
        ("Q15514380", "Margaret McMillan"),
        ("Q15451133", "Margaret Olrog Stoddart"),
        ("Q6760074", "Margaret Preston"),
        ("Q19803013", "Margaret Sarah Geddes"),
        ("Q16011996", "Margaret Thomas"),
        ("Q6760110", "Margaret Tarrant"),
        ("Q87649701", "Margaret Tod"),
        ("Q19803016", "Margarete Schall"),
        ("Q18091983", "Margaretha Roosenboom"),
        ("Q20980049", "Margarett Sargent"),
        ("Q1395151", "Margherita Barezzi"),
        ("Q21466972", "Margherita Gonzaga"),
        
        # Pacific Island Artists
        ("Q1111994", "Marguerite De Angeli"),
        ("Q3290532", "Marguerite Dufay"),
        ("Q16066084", "Marguerite Kirmse"),
        ("Q1111998", "Marguerite Stuber Pearson"),
        ("Q16190013", "Marguerite Thompson Zorach"),
        ("Q15998925", "Maria Brooks"),
        ("Q16089598", "Maria Euphrosyne Spartali"),
        ("Q3292416", "Maria Germana Messaggi"),
        ("Q6761230", "Maria Innocentia Hummel"),
        ("Q18577013", "Maria Martin"),
        ("Q3292419", "Maria Oakey Dewing"),
        ("Q23905632", "Maria Spartali Stillman"),
        ("Q3847207", "Maria Wiik"),
        ("Q18559873", "Marian Ellis Rowan"),
        ("Q20980051", "Marian Emma Chase"),
        ("Q21456911", "Marianna Slater"),
        ("Q19999938", "Marianne Preindelsberger Stokes"),
        ("Q6762836", "Marianne von Werefkin"),
        
        # Indigenous Artists
        ("Q1112075", "Marie Aimée Lucas-Robiquet"),
        ("Q20002047", "Marie Cazin"),
        ("Q3292576", "Marie Danforth Page"),
        ("Q21452012", "Marie de Rabutin-Chantal"),
        ("Q15514384", "Marie Duhem"),
        ("Q21451316", "Marie Elisabeth Wiegmann"),
        ("Q15514386", "Marie Krøyer"),
        ("Q17626245", "Marie Lucas-Robiquet"),
        ("Q15972107", "Marie Petiet"),
        ("Q21281827", "Marie Spartali Stillman"),
        ("Q15514388", "Marie Triepcke Krøyer Alfvén"),
        ("Q21451318", "Marie-Amélie Cogniet"),
        ("Q22915293", "Marie-Christine de Habsbourg-Lorraine"),
        ("Q15972109", "Marie-Ernestine Serret"),
        ("Q3292581", "Marie-Françoise Constance Mayer"),
        ("Q21995181", "Marie-Louise Petiet"),
        ("Q16028116", "Marie-Paule Deville-Chabrolle"),
        ("Q18510058", "Marie-Rosalie Bonheur"),
        
        # South Asian Artists
        ("Q16066088", "Marietta Robusti"),
        ("Q21281821", "Marina Sainte-Catherine"),
        ("Q6764021", "Marion Boyd Allen"),
        ("Q3295028", "Marion Greenwood"),
        ("Q18020846", "Marion Kavanaugh Wachtel"),
        ("Q18020848", "Marion Post Wolcott"),
        ("Q18976570", "Mariquita Jenny Moberly"),
        ("Q21402275", "Marjorie Acker"),
        ("Q23640316", "Marjorie Content"),
        ("Q1112171", "Marjorie Organ"),
        ("Q6766325", "Marjorie Strider"),
        ("Q3295251", "Marthe Berard"),
        ("Q21557960", "Martha Baker"),
        ("Q6774560", "Martha Cahoon"),
        ("Q19803037", "Martha Darley Mutrie"),
        ("Q6774648", "Martha Susan Baker"),
        ("Q16089625", "Martha Walter"),
        ("Q1899788", "Mary Agnes Yerkes"),
        ("Q6778924", "Mary Alcott"),
        
        # Southeast Asian Artists
        ("Q17501887", "Mary Allen"),
        ("Q21281823", "Mary Ann Slater"),
        ("Q6778931", "Mary Beale"),
        ("Q20899871", "Mary Bradish Titcomb"),
        ("Q15705406", "Mary Cameron"),
        ("Q16066097", "Mary Chauncey"),
        ("Q27571224", "Mary Creese"),
        ("Q21002465", "Mary Davis"),
        ("Q19999951", "Mary Dignam"),
        ("Q16066098", "Mary Fairchild MacMonnies Low"),
        ("Q18684765", "Mary Fraser Tytler"),
        ("Q21451328", "Mary Gay"),
        ("Q27889977", "Mary Georgina Barton"),
        ("Q16028149", "Mary Grant"),
        ("Q18020857", "Mary Hallock Foote"),
        ("Q1906841", "Mary Heilmann"),
        ("Q21466988", "Mary Hutchinson"),
        ("Q21995220", "Mary Kessell"),
        
        # Central Asian Artists
        ("Q18020858", "Mary Knight"),
        ("Q15514400", "Mary L. Macomber"),
        ("Q6779863", "Mary Lloyd"),
        ("Q6779879", "Mary Louise McLaughlin"),
        ("Q6779920", "Mary MacMonnies"),
        ("Q6779930", "Mary McCroskey"),
        ("Q16190090", "Mary McEvoy"),
        ("Q6779968", "Mary Moser"),
        ("Q20001451", "Mary Morris"),
        ("Q6780018", "Mary Nicol Neill Armour"),
        ("Q18020860", "Mary Nimmo Moran"),
        ("Q16066100", "Mary Osborn"),
        ("Q14593666", "Mary Prince"),
        ("Q6780189", "Mary Rogers Williams"),
        ("Q16028171", "Mary Russell"),
        ("Q22669722", "Mary Sargant Florence"),
        ("Q94629831", "Mary Shepard Greene Blumenschein"),
        ("Q16066101", "Mary Snell Pringle"),
        
        # North African Artists
        ("Q21456913", "Mary Swanzy"),
        ("Q6780421", "Mary Tannahill"),
        ("Q21002462", "Mary Thornycroft"),
        ("Q20002049", "Mary Vaux Walcott"),
        ("Q6780651", "Mary Watts"),
        ("Q15451265", "Mary Way"),
        ("Q21451322", "Mary Willumsen"),
        ("Q21456917", "Maryse Bastié"),
        ("Q517189", "Mathilde Blind"),
        ("Q517192", "Mathilde Bonaparte"),
        ("Q18020864", "Mathilde Mueden"),
        ("Q6787526", "Matilda Browne"),
        ("Q18020866", "Matilda Lotz"),
        ("Q21289361", "Maud Earl"),
        ("Q6792523", "Maud Lewis"),
        ("Q21466984", "Maude Adams"),
        ("Q21452051", "May Alcott Nieriker"),
        ("Q15514402", "May Morris"),
        
        # Contemporary International Artists
        ("Q20980045", "Mechthild Lang"),
        ("Q23020053", "Mednyánszky Cécile"),
        ("Q6816522", "Melvina Hoffman"),
        ("Q21451333", "Merete Möller"),
        ("Q18020869", "Meta Vaux Warrick Fuller"),
        ("Q16053468", "Mildred Anne Butler"),
        ("Q1112326", "Mildred Bailey"),
        ("Q6851033", "Mildred Holland"),
        ("Q16859960", "Millicent Rogers"),
        ("Q18761877", "Mina Carlson-Bredberg"),
        ("Q18559996", "Mina Fonda Ochtman"),
        ("Q6862164", "Mina Loy"),
        ("Q18510128", "Minerva Chapman"),
        ("Q18510129", "Minerva J. Chapman"),
        ("Q21451344", "Minnie Ashley"),
        ("Q15514405", "Minnie Evans"),
        ("Q18020875", "Minnie Harms Neebe"),
        ("Q18020877", "Minna Canth"),
        ("Q21002558", "Miriam Pearse"),
        ("Q232423", "Frida Kahlo"),
        ("Q204832", "Mary Cassatt"),
        ("Q260351", "Berthe Morisot"),
        ("Q7836", "Marie-Louise-Élisabeth Vigée-Lebrun"),
        ("Q235928", "Georgia O'Keeffe"),
        ("Q451652", "Angelica Kauffman"),
        ("Q231027", "Judith Leyster"),
        ("Q237816", "Rosa Bonheur"),
        ("Q233482", "Suzanne Valadon"),
        ("Q467703", "Rachel Ruysch"),
        ("Q433989", "Tamara de Lempicka"),
        ("Q272320", "Helen Frankenthaler"),
        ("Q239865", "Joan Mitchell"),
        ("Q231136", "Eva Gonzalès"),
        ("Q2248775", "Sofonisba Anguissola"),
        ("Q1362683", "Lavinia Fontana"),
        ("Q241732", "Clara Peeters"),
        ("Q270572", "Marie Laurencin"),
        ("Q47146", "Käthe Kollwitz"),
        ("Q242068", "Paula Modersohn-Becker"),
        ("Q255995", "Hilma af Klint"),
        ("Q236875", "Rosalba Carriera"),
        ("Q264351", "Marianne North"),
        ("Q260054", "Zinaida Serebriakova"),
        ("Q232265", "Natalia Goncharova"),
        ("Q469981", "Maria Sibylla Merian"),
        ("Q229241", "Fede Galizia"),
        ("Q450848", "Giovanna Garzoni"),
        ("Q3896498", "Plautilla Nelli"),
        ("Q2347694", "Elisabetta Sirani"),
        ("Q233794", "Vanessa Bell"),
        ("Q435316", "Gwen John"),
        ("Q232407", "Emily Carr"),
        ("Q232644", "Anna Ancher"),
        ("Q273579", "Marie Bracquemond"),
        ("Q437232", "Marlene Dumas"),
        ("Q432948", "Julie Mehretu"),
        ("Q441459", "Kara Walker"),
        ("Q264132", "Bridget Riley"),
        ("Q237442", "Agnes Martin"),
        ("Q460188", "Yayoi Kusama"),
        ("Q230203", "Cindy Sherman"),
        ("Q441147", "Jenny Saville"),
        ("Q233191", "Tracey Emin"),
         ("Q3684006", "Caterina van Hemessen"),
        ("Q2837607", "Levina Teerlinc"),
        ("Q3051640", "Ellen Wallace Sharples"),
        ("Q232481", "Louise Abbéma"),
        ("Q460408", "Harriet Backer"),
        ("Q437648", "Laura Muntz Lyall"),
        ("Q290492", "Elisabeth Jerichau-Baumann"),
        ("Q266063", "Louise Breslau"),
        ("Q1906681", "Marianne von Werefkin"),
        ("Q273206", "Florine Stettheimer"),
        ("Q449986", "Romaine Brooks"),
        ("Q3602216", "Abigail de Andrade"),
        ("Q450295", "Alice Neel"),
        ("Q450308", "Lee Krasner"),
        ("Q538011", "Élisabeth Louise Vigée Le Brun"),
        ("Q3482654", "Properzia de' Rossi"),
        ("Q464130", "Katharina van Hemessen"),
        ("Q276032", "Anna Dorothea Therbusch"),
        ("Q456903", "Angelika Platner"),
        ("Q444915", "Edmonia Lewis"),
        ("Q461907", "Lilla Cabot Perry"),
        ("Q469516", "Cecilia Beaux"),
        ("Q233718", "Marie Bashkirtseff"),
        ("Q439608", "Eva Bonnier"),
        ("Q273177", "Helene Schjerfbeck"),
        ("Q441090", "Jacqueline Marval"),
        ("Q456226", "Ida Gerhardi"),
        ("Q460211", "Gabriele Münter"),
        ("Q271339", "Tarsila do Amaral"),
        ("Q3089012", "Marie-Guillemine Benoist"),
        ("Q276198", "Constance Mayer"),
        ("Q452316", "Louise-Joséphine Sarazin de Belmont"),
        ("Q437994", "Adélaïde Labille-Guiard"),
        ("Q3170098", "Jeanne-Elisabeth Chaudet"),
        ("Q3605504", "Adrienne Marie Louise Grandpierre-Deverzy"),
        ("Q232616", "Sophie Anderson"),
        ("Q439124", "Kate Perugini"),
        ("Q543173", "Evelyn De Morgan"),
        ("Q232402", "Eleanor Fortescue-Brickdale"),
        ("Q4349638", "Marie Spartali Stillman"),
        ("Q260681", "Rebecca Solomon"),
        ("Q450317", "Elizabeth Siddal"),
        ("Q455788", "Emma Sandys"),
        ("Q434742", "Julia Margaret Cameron"),
        ("Q271084", "Frances Hodgkins"),
        ("Q271669", "Rita Angus"),
        ("Q444543", "Grace Cossington Smith"),
        ("Q460219", "Margaret Preston"),
        ("Q2298515", "Clarice Beckett"),
        ("Q455214", "Nora Heysen"),
        ("Q1350954", "Kathleen O'Connor"),
        ("Q2399659", "Bessie Davidson"),
        ("Q449785", "Thea Proctor"),
        ("Q6376218", "Kathleen Morris"),
        ("Q445351", "Doris Boyd"),
        ("Q438258", "Joy Hester"),
        ("Q262234", "Sidney Nolan"),
        ("Q231186", "Marie Ellenrieder"),
        ("Q538015", "Louise Seidler"),
        ("Q449612", "Caroline Bardua"),
        ("Q450366", "Maria Electrine von Freyberg"),
        ("Q456750", "Barbara Krafft"),
        ("Q273563", "Vilma Parlaghy"),
        ("Q233456", "Hermine David"),
        ("Q232515", "Marie Laurencin"),
        ("Q450350", "Jacqueline Lamba"),
        ("Q232596", "Kay Sage"),
        ("Q260288", "Leonor Fini"),
        ("Q238045", "Remedios Varo"),
        ("Q441557", "Leonora Carrington"),
        ("Q260529", "Dorothea Tanning"),
        ("Q469930", "Toyen"),
        ("Q3950640", "Sarah Purser"),
        ("Q458965", "Mainie Jellett"),
        ("Q1897219", "Mary Swanzy"),
        ("Q538009", "Evie Hone"),
        ("Q6780556", "Norah McGuinness"),
        ("Q538012", "Grace Henry"),
        ("Q289490", "Wilhelmina Geddes"),
        ("Q4821455", "Beatrice Elvery"),
        ("Q449704", "Katherine Sophie Dreier"),
        ("Q457190", "Florine Stettheimer"),
        ("Q450301", "Marguerite Zorach"),
        ("Q439677", "Henrietta Shore"),
        ("Q532439", "Marjorie Acker Phillips"),
        ("Q20200253", "Jane Peterson"),
        ("Q461903", "Bessie Potter Vonnoh"),
        ("Q461185", "Anna Hyatt Huntington"),
        ("Q3057913", "Ethel Walker"),
        ("Q458906", "Laura Knight"),
        ("Q438789", "Gwen Raverat"),
        ("Q456656", "Dora Carrington"),
        ("Q458984", "Jessica Dismorr"),
        ("Q1395875", "Helen Saunders"),
        ("Q6136528", "Winifred Knights"),
        ("Q2836374", "Gladys Hynes"),
        ("Q3359379", "Paule Vézelay"),
        ("Q460577", "Eileen Agar"),
        ("Q450259", "Ithell Colquhoun"),
        ("Q441721", "Wilhelmina Barns-Graham"),
        ("Q19667503", "Marlow Moss"),
        ("Q434763", "Margaret Macdonald Mackintosh"),
        ("Q4821458", "Frances MacDonald"),
        ("Q453972", "Jessie M. King"),
        ("Q538018", "Ann Macbeth"),
        ("Q455206", "Helen McNicoll"),
        ("Q440975", "Emily Mary Osborn"),
        ("Q445086", "Sophie Gengembre Anderson"),
        ("Q458099", "Elizabeth Thompson"),
        ("Q265423", "Henriette Browne"),
        ("Q444594", "Rosa Brett"),
        ("Q3603985", "Adriana Johanna Haanen"),
        ("Q1897214", "Margareta Haverman"),
        ("Q454972", "Maria van Oosterwijck"),
        ("Q3292417", "Maria Schalcken"),
        ("Q21647166", "Gesina ter Borch"),
        ("Q3108626", "Cornelia van der Mijn"),
        ("Q2941198", "Catharina Backer"),
        ("Q21451072", "Aleijda Wolfsen"),
        ("Q448016", "Josefa de Óbidos"),
        ("Q263196", "Luisa Roldán"),
        ("Q455768", "Josefa de Ayala"),
        ("Q9010340", "María de Guadalupe"),
        ("Q437761", "Catharina van Hemessen"),
        ("Q233441", "Diana Scultori"),
        ("Q1241673", "Marietta Robusti"),
        ("Q3497656", "Barbara Longhi"),
        ("Q435927", "Lucia Anguissola"),
        ("Q2836369", "Elena Anguissola"),
        ("Q2836370", "Europa Anguissola"),
        ("Q3820187", "Anna Maria Anguissola"),
        ("Q3850195", "Minerva Anguissola"),
        ("Q435903", "Maria Ormani"),
        ("Q3292418", "Antonia Uccello"),
        ("Q437463", "Onorata Rodiana"),
        ("Q3605061", "Sister Plautilla"),
        ("Q441525", "Catherine of Bologna"),
        ("Q458762", "Claricia"),
        ("Q469987", "Ende"),
        ("Q11923632", "Guda"),
        ("Q463824", "Diemudis"),
        ("Q27924340", "Herrad of Landsberg"),
        ("Q257161", "Hildegard of Bingen"),
        ("Q451305", "Teresa Díez"),
        ("Q438361", "María Blanchard"),
        ("Q461652", "Ángeles Santos Torroella"),
        ("Q460556", "Maruja Mallo"),
        ("Q445150", "Delhy Tejero"),
        ("Q5951103", "Menchu Gal"),
        ("Q431731", "Carmen Laffón"),
        ("Q272603", "María Moreno"),
        ("Q1908042", "Soledad Sevilla"),
        ("Q460176", "Ouka Leele"),
        ("Q433869", "Cristina Iglesias"),
        ("Q276104", "Susy Gómez"),
        ("Q455380", "Elena Asins"),
        ("Q441989", "Amalia Avia"),
        ("Q20014607", "Isabel Quintanilla"),
        ("Q5951605", "Esperanza Parada"),
        ("Q449674", "Anna Bilińska-Bohdanowicz"),
        ("Q449813", "Olga Boznańska"),
        ("Q260599", "Zofia Stryjeńska"),
        ("Q437449", "Tamara Łempicka"),
        ("Q274579", "Maria Jarema"),
        ("Q11768291", "Alina Szapocznikow"),
        ("Q271851", "Magdalena Abakanowicz"),
        ("Q450304", "Katarzyna Kobro"),
        ("Q11715665", "Teresa Żarnowerówna"),
        ("Q437339", "Fannie Moody"),
        ("Q437574", "Anna Lea Merritt"),
        ("Q450297", "Marie Bracquemond"),
        ("Q231360", "Victorine Meurent"),
        ("Q266039", "Louise Catherine Breslau"),
        ("Q463866", "Virginie Demont-Breton"),
        ("Q435708", "Henriette Lorimier"),
        ("Q439077", "Jeanne-Philiberte Ledoux"),
        ("Q272982", "Marguerite Gérard"),
        ("Q3292416", "Marie-Victoire Lemoine"),
        ("Q441232", "Marie-Geneviève Bouliar"),
        ("Q3089011", "Marie-Gabrielle Capet"),
        ("Q456910", "Nanine Vallain"),
        ("Q2837111", "Herminie Gudin"),
        ("Q458638", "Marie-Éléonore Godefroid"),
        ("Q3292414", "Eugénie Servières"),
        ("Q438794", "Félicie de Fauveau"),
        ("Q456907", "Marie d'Orléans"),
        ("Q441305", "Nélie Jacquemart"),
        ("Q272530", "Louise-Élisabeth de Meuron"),
        ("Q458641", "Henriette Lorimier"),
        ("Q467268", "Julie Volpelière"),
        ("Q456915", "Jeanne Bole du Chomont"),
        ("Q467892", "Clémence Roth"),
        ("Q1232600", "Anna Palm de Rosa"),
        ("Q469639", "Sofia Adlersparre"),
        ("Q4946939", "Amalia Lindegren"),
        ("Q4936055", "Agnes Börjesson"),
        ("Q4945456", "Lea Ahlborn"),
        ("Q4935232", "Amanda Sidwall"),
        ("Q4936068", "Josefina Holmlund"),
        ("Q291550", "Jenny Nyström"),
        ("Q4935236", "Fanny Brate"),
        ("Q272581", "Sigrid Hjertén"),
        ("Q297937", "Vera Nilsson"),
        ("Q4946938", "Ninnan Santesson"),
        ("Q4946940", "Siri Derkert"),
        ("Q4945290", "Mollie Faustman"),
        ("Q451084", "Berta Hansson"),
        ("Q4936055", "Maj Bring"),
        ("Q435721", "Ulrica Hydman-Vallien"),
        ("Q456901", "Lena Cronqvist"),
        ("Q433984", "Karin Mamma Andersson"),
        ("Q433989", "Annika von Hausswolff"),
        ("Q450305", "Nathalia Edenmont"),
        ("Q433873", "Klara Lidén"),
        ("Q7251305", "Faith Ringgold"),
        ("Q460447", "Alma Thomas"),
        ("Q440763", "Lois Mailou Jones"),
        ("Q455316", "Elizabeth Catlett"),
        ("Q1908195", "Augusta Savage"),
        ("Q461427", "Laura Wheeler Waring"),
        ("Q437214", "Meta Vaux Warrick Fuller"),
        ("Q455210", "Edmonia Lewis"),
        ("Q15060026", "Clementine Hunter"),
        ("Q5433797", "Minnie Evans"),
        ("Q442975", "Sister Gertrude Morgan"),
        ("Q435725", "Howardena Pindell"),
        ("Q435725", "Emma Amos"),
        ("Q455382", "Betye Saar"),
        ("Q439564", "Mickalene Thomas"),
        ("Q439556", "Amy Sherald"),
        ("Q21176592", "Kehinde Wiley"),
        ("Q5569953", "Njideka Akunyili Crosby"),
        ("Q464614", "Lubaina Himid"),
        ("Q460447", "Lynette Yiadom-Boakye"),
        ("Q450379", "Sonia Boyce"),
        ("Q435725", "Veronica Ryan"),
        ("Q439606", "Hurvin Anderson"),
        ("Q450379", "Claudette Johnson"),
        ("Q460182", "Khadija Saye"),
        ("Q437791", "Anthea Hamilton"),
        ("Q435928", "Gillian Wearing"),
        ("Q271820", "Sarah Lucas"),
        ("Q434303", "Rachel Whiteread"),
        ("Q469659", "Fiona Rae"),
        ("Q1282173", "Jenny Saville"),
        ("Q445332", "Cornelia Parker"),
        ("Q435928", "Tacita Dean"),
        ("Q435726", "Mona Hatoum"),
        ("Q272601", "Phyllida Barlow"),
        ("Q269649", "Rose Wylie"),
        ("Q437991", "Maggi Hambling"),
        ("Q447547", "Paula Rego"),
        ("Q445458", "Lubaina Himid"),
        ("Q466167", "Eileen Cooper"),
        ("Q456901", "Sonia Delaunay"),
        ("Q467856", "Marie Vassilieff"),
        ("Q440239", "Natalia Dumitresco"),
        ("Q437226", "Maria Helena Vieira da Silva"),
        ("Q274230", "Aurelie Nemours"),
        ("Q260534", "Françoise Gilot"),
        ("Q456890", "Marie Raymond"),
        ("Q266093", "Marcelle Cahn"),
        ("Q456877", "Vera Molnár"),
        ("Q1820137", "Geneviève Asse"),
        ("Q271862", "Sophie Taeuber-Arp"),
        ("Q456650", "Meret Oppenheim"),
        ("Q463039", "Miriam Cahn"),
        ("Q11694088", "Pipilotti Rist"),
        ("Q450308", "Hannah Höch"),
        ("Q456391", "Jeanne Mammen"),
        ("Q461897", "Marianne Brandt"),
        ("Q450379", "Anni Albers"),
        ("Q445087", "Gunta Stölzl"),
        ("Q97573", "Benita Koch-Otte"),
        ("Q455798", "Gertrud Arndt"),
        ("Q57412", "Lotte Stam-Beese"),
        ("Q271820", "Inge Scholl"),
        ("Q57296", "Lucia Moholy"),
        ("Q450308", "Lotte Jacobi"),
        ("Q450305", "Ellen Auerbach"),
        ("Q437708", "Ilse Bing"),
        ("Q441481", "Grete Stern"),
        ("Q450312", "Germaine Krull"),
        ("Q437703", "Florence Henri"),
        ("Q64855", "Berenice Abbott"),
        ("Q437648", "Margaret Bourke-White"),
        ("Q232524", "Dorothea Lange"),
        ("Q441806", "Imogen Cunningham"),
        ("Q439381", "Ruth Bernhard"),
        ("Q442568", "Lisette Model"),
        ("Q537002", "Helen Levitt"),
        ("Q231588", "Diane Arbus"),
        ("Q438759", "Vivian Maier"),
        ("Q466512", "Francesca Woodman"),
        ("Q450305", "Sally Mann"),
        ("Q450317", "Nan Goldin"),
        ("Q456901", "Shirin Neshat"),
        ("Q256434", "Marina Abramović"),
        ("Q232543", "Carolee Schneemann"),
        ("Q235034", "Ana Mendieta"),
        ("Q459462", "Adrian Piper"),
        ("Q461185", "Hannah Wilke"),
        ("Q433984", "Eleanor Antin"),
        ("Q437756", "Martha Rosler"),
        ("Q467856", "Mary Kelly"),
        ("Q234370", "Judy Chicago"),
        ("Q456387", "Miriam Schapiro"),
        ("Q451048", "Joyce Kozloff"),
        ("Q456910", "Nancy Spero"),
        ("Q437684", "May Stevens"),
        ("Q276555", "Sylvia Sleigh"),
        ("Q450317", "Joan Semmel"),
        ("Q538018", "Audrey Flack"),
        ("Q451232", "Janet Fish"),
        ("Q6776037", "Martha Diamond"),
        ("Q450398", "Susan Rothenberg"),
        ("Q450256", "Jennifer Bartlett"),
        ("Q15873778", "Elizabeth Murray"),
        ("Q448570", "Katherine Bradford"),
        ("Q437695", "Nicole Eisenman"),
        ("Q437989", "Dana Schutz"),
        ("Q433989", "Amy Sillman"),
        ("Q451345", "Laura Owens"),
        ("Q21282812", "Katherine Bernhardt"),
        ("Q2836157", "Charline von Heyl"),
        ("Q450397", "Mari Eastman"),
        ("Q433987", "Lisa Yuskavage"),
        ("Q439554", "Cecily Brown"),
        ("Q437256", "Elizabeth Peyton"),
        ("Q434303", "Rita Ackermann"),
        ("Q450312", "Katharina Fritsch"),
        ("Q55433508", "Rosemarie Trockel"),
        ("Q450318", "Isa Genzken"),
        ("Q164746", "Rebecca Horn"),
        ("Q439479", "Hanne Darboven"),
        ("Q455204", "Maria Lassnig"),
        ("Q450305", "Valie Export"),
        ("Q445089", "Martha Jungwirth"),
        ("Q269649", "Elke Krystufek"),
        ("Q451337", "Eva Schlegel"),
        ("Q439556", "Kiki Kogelnik"),
        ("Q538009", "Maria Hahnenkamp"),
        ("Q441482", "Birgit Jürgenssen"),
        ("Q78661", "Renate Bertlmann"),
        ("Q456390", "VALIE EXPORT"),
        ("Q451283", "Geta Brătescu"),
        ("Q12720050", "Lia Perjovschi"),
        ("Q6241286", "Alina Szapocznikow"),
        ("Q271869", "Natalia LL"),
        ("Q11715668", "Ewa Partum"),
        ("Q7834302", "Zofia Kulik"),
        ("Q11768019", "Krzysztof Wodiczko"),
        ("Q9368838", "Teresa Murak"),
        ("Q439555", "Anna Bella Geiger"),
        ("Q2625639", "Lygia Clark"),
        ("Q441422", "Lygia Pape"),
        ("Q456856", "Mira Schendel"),
        ("Q449677", "Anna Maria Maiolino"),
        ("Q3821009", "Letícia Parente"),
        ("Q437226", "Regina Silveira"),
        ("Q20977662", "Rosângela Rennó"),
        ("Q449998", "Beatriz Milhazes"),
        ("Q449702", "Adriana Varejão"),
        ("Q538003", "Jac Leirner"),
        ("Q15407288", "Fernanda Gomes"),
        ("Q5925593", "Lucia Laguna"),
        ("Q539669", "Laura Lima"),
        ("Q20015097", "Rivane Neuenschwander"),
        ("Q11336334", "Cao Fei"),
        ("Q700427", "Lin Tianmiao"),
        ("Q6548549", "Yin Xiuzhen"),
        ("Q11303557", "Xiao Lu"),
        ("Q271869", "He Chengyao"),
        ("Q16149932", "Chen Lingyang"),
        ("Q15954954", "Ma Liuming"),
        ("Q700425", "Cui Xiuwen"),
        ("Q11618995", "Peng Yu"),
        ("Q437701", "Liu Wei"),
        ("Q703207", "Yu Hong"),
        ("Q451299", "Li Jin"),
        ("Q15453831", "Geng Xue"),
        ("Q6786936", "Muzi Mei"),
        ("Q707040", "Kan Xuan"),
        ("Q445642", "Vigée Le Brun"),
        ("Q3063427", "Marguerite de Valois"),
        ("Q457062", "Marie-Anne Collot"),
        ("Q3296038", "Marie-Suzanne Giroust"),
        ("Q274217", "Françoise Duparc"),
        ("Q458059", "Anne Vallayer-Coster"),
        ("Q3816624", "Aimée Brune"),
        ("Q451742", "Henriette Rath"),
        ("Q19832712", "Cesarine Davin-Mirvault"),
        ("Q2941198", "Marie-Denise Villers"),
        ("Q456620", "Pauline Auzou"),
        ("Q3089595", "Sophie Rude"),
        ("Q3296099", "Hortense Haudebourt-Lescot"),
        ("Q458053", "Félicie de Fauveau"),
        ("Q3820436", "Léonie Matthis"),
        ("Q3089015", "Constance Charpentier"),
        ("Q273328", "Charlotte Bonaparte"),
        ("Q3046192", "Eulalie Morin"),
        ("Q456754", "Henriette Cappelaere"),
        ("Q16669708", "Amélie Serre"),
        ("Q4775114", "Antoinette Haudebourt-Lescot"),
        ("Q19606504", "Julie Duvidal de Montferrier"),
        ("Q3296126", "Joséphine Calamatta"),
        ("Q3820998", "Louise Vernet"),
        ("Q3188744", "Julie Ribault"),
        ("Q2934981", "Camille Roqueplan"),
        ("Q16028935", "Clémence Naigeon"),
        ("Q21457768", "Eugénie Dalton"),
        ("Q15970982", "Laure Devéria"),
        ("Q3220015", "Léocadie Doze"),
        ("Q28018903", "Palmyre Granger"),
        ("Q3371494", "Pauline Garon"),
        ("Q19606503", "Sophie Liénard"),
        ("Q108170434", "Valentine Reyre"),
        ("Q21395209", "Zoé Laure de Chatillon"),
        ("Q19544620", "Adèle Riché"),
        ("Q21282639", "Augustine Dufresne"),
        ("Q27867539", "Bathilde de Chateaubourg"),
        ("Q17177615", "Caroline Wietzel"),
        ("Q16859860", "Charlotte Soyer"),
        ("Q108170433", "Ernestine Friedmann"),
        ("Q19606468", "Fanny Beauharnais"),
        ("Q15970916", "Félicité Lagarenne"),
        ("Q16028849", "Henriette Raymond"),
        ("Q3051095", "Elisabeth Sonrel"),
        ("Q18508662", "Marie-Adélaïde Durieux"),
        ("Q3820414", "Laure Brouardel"),
        ("Q94564924", "Amélie Lundahl"),
        ("Q18576624", "Bertha Wegmann"),
        ("Q4948440", "Emilie Mundt"),
        ("Q273327", "Kitty Lange Kielland"),
        ("Q15998842", "Oda Krohg"),
        ("Q11881678", "Astri Welhaven Heiberg"),
        ("Q4936053", "Hildur Söderberg"),
        ("Q15451195", "Ellen Jolin"),
        ("Q4936067", "Elsa Beskow"),
        ("Q270566", "Ottilia Adelborg"),
        ("Q4355323", "Tekla Swedlund"),
        ("Q20685918", "Esther Kjerner"),
        ("Q4947495", "Hanna Hirsch-Pauli"),
        ("Q15451196", "Eva Bagge"),
        ("Q4768669", "Anna Boberg"),
        ("Q4933943", "Wilhelmina Lagerholm"),
        ("Q4936042", "Julia Beck"),
        
        # Dutch and Flemish Painters  
        ("Q14405823", "Sara van Baalbergen"),
        ("Q2710028", "Alida Pott"),
        ("Q1904668", "Thérèse Schwartze"),
        ("Q2343981", "Sientje Mesdag-van Houten"),
        ("Q467906", "Suze Robertson"),
        ("Q458542", "Charley Toorop"),
        ("Q441333", "Else Berg"),
        ("Q1968308", "Jacoba van Heemskerck"),
        ("Q14639893", "Coba Ritsema"),
        ("Q2801893", "Ans van den Berg"),
        ("Q2344008", "Jo Koster"),
        ("Q15879855", "Lizzy Ansingh"),
        ("Q2801824", "Elsa Bakalar"),
        ("Q3132659", "Marie Henrie Mackenzie"),
        ("Q18508685", "Henriëtte Ronner-Knip"),
        ("Q21550675", "Marie Wandscheer"),
        ("Q2488358", "Betsy Perk"),
        ("Q2801831", "Aletta de Frey"),
        ("Q2710000", "Anna Abrahams"),
        
        # German Speaking Artists
        ("Q15451298", "Maria Slavona"),
        ("Q19961268", "Elfriede Lohse-Wächtler"),
        ("Q1397223", "Ida Gerhardi"),
        ("Q1518384", "Clara Rilke-Westhoff"),
        ("Q214539", "Else Lasker-Schüler"),
        ("Q107454", "Emmy Hennings"),
        ("Q456832", "Renée Sintenis"),
        ("Q106567", "Lotte Reiniger"),
        ("Q93761", "Elfriede Lohse-Wächtler"),
        ("Q71396", "Martel Schwichtenberg"),
        ("Q451515", "Elisabeth Erdmann-Macke"),
        ("Q451516", "Maria Marc"),
        ("Q456815", "Nell Walden"),
        ("Q451517", "Emmy Klinker"),
        ("Q451518", "Marta Worringer"),
        ("Q455831", "Emy Roeder"),
        ("Q1395114", "Martha Burkhardt"),
        ("Q451522", "Emmy Roth"),
        ("Q94657", "Anita Rée"),
        
        # Austrian Artists
        ("Q2632512", "Elena Luksch-Makowsky"),
        ("Q451524", "Emilie Mediz-Pelikan"),
        ("Q16016806", "Teresa Feodorowna Ries"),
        ("Q1517893", "Marie Egner"),
        ("Q451527", "Tina Blau"),
        ("Q19006779", "Olga Wisinger-Florian"),
        ("Q1762848", "Broncia Koller-Pinell"),
        ("Q2632515", "Helene Funke"),
        ("Q18508640", "Fritzi Löw"),
        ("Q7925529", "Stephanie Glax"),
        ("Q15451356", "Mileva Roller"),
        ("Q15845819", "Grete Wolf Krakauer"),
        ("Q1553110", "Gudrun Baudisch"),
        ("Q1397224", "Erika Giovanna Klien"),
        ("Q451531", "Friedl Dicker-Brandeis"),
        ("Q451532", "Carry Hauser"),
        ("Q1112333", "Susanne Wenger"),
        ("Q16016814", "Arik Brauer"),
        
        # Belgian Artists
        ("Q16853010", "Anna Boch"),
        ("Q2851693", "Anna De Weert"),
        ("Q4768013", "Berthe Art"),
        ("Q15071932", "Louise Danse"),
        ("Q18508764", "Marie Collart"),
        ("Q3318612", "Yvonne Serruys"),
        ("Q21654728", "Alice Ronner"),
        ("Q21654729", "Emma Ronner"),
        ("Q21654730", "Mathilde Ronner"),
        ("Q16674035", "Euphrosine Beernaert"),
        ("Q3194851", "Ketty de la Rocque-Sevrin"),
        ("Q21654731", "Marthe Massin"),
        ("Q21654732", "Marthe Donas"),
        ("Q21654733", "Jane Graverol"),
        ("Q21654734", "Rachel Baes"),
        ("Q7885085", "Suzanne Thienpont"),
        
        # Swiss Artists
        ("Q117366", "Alice Bailly"),
        ("Q123078", "Martha Stettler"),
        ("Q2831305", "Clara von Rappard"),
        ("Q15451303", "Louise-Cathérine Breslau"),
        ("Q3950641", "Jeanne Lombard"),
        ("Q21654735", "Marcelle Schinz"),
        ("Q15451304", "Regina Conti"),
        ("Q15451305", "Anne-Marie von Matt"),
        ("Q125353", "Emma Kunz"),
        ("Q439547", "Miriam Cahn"),
        ("Q118845", "Niki de Saint Phalle"),
        
        # Italian Artists
        ("Q3714606", "Sofia di Chiara Marucelli"),
        ("Q3633650", "Bice Lazzari"),
        ("Q3633651", "Carla Accardi"),
        ("Q3633652", "Dadamaino"),
        ("Q3633653", "Carol Rama"),
        ("Q3633654", "Titina Maselli"),
        ("Q23901911", "Antonietta Raphaël"),
        ("Q3633655", "Renata Bonfanti"),
        ("Q532439", "Benedetta Cappa"),
        ("Q3633656", "Fausta Squatriti"),
        ("Q461922", "Carla Badiali"),
        ("Q3633657", "Grazia Varisco"),
        ("Q16561536", "Tomaso Binga"),
        ("Q3633658", "Ketty La Rocca"),
        ("Q16561537", "Nanda Vigo"),
        ("Q3633659", "Giosetta Fioroni"),
        
        # Spanish Artists
        ("Q5888695", "María Roësset Mosquera"),
        ("Q6003315", "Julia Minguillón"),
        ("Q9015451", "Lluïsa Vidal"),
        ("Q12388694", "Pepita Teixidor"),
        ("Q11685318", "Antonia Ferreras"),
        ("Q16940454", "Elvira Santamaría"),
        ("Q22676793", "Laura Albéniz"),
        ("Q11928820", "María Corredoira"),
        ("Q9061179", "Rosario de Velasco"),
        ("Q11697116", "Juana Romani"),
        ("Q15973654", "Fernanda Francés"),
        ("Q18699567", "Adela Ginés"),
        ("Q19519168", "Elena Brockmann"),
        ("Q5836491", "Lola Anglada"),
        ("Q11941181", "Victorina Durán"),
        
        # Portuguese Artists
        ("Q10301819", "Josefa de Óbidos"),
        ("Q10346838", "Maria Keil"),
        ("Q10328208", "Mily Possoz"),
        ("Q10302280", "Vieira da Silva"),
        ("Q10302281", "Sarah Affonso"),
        ("Q10302282", "Ofélia Marques"),
        ("Q10302283", "Alice Rey Colaço"),
        ("Q10302284", "Estrela Faria"),
        ("Q10302285", "Maria Helena Matos"),
        ("Q10302286", "Menez"),
        ("Q10302287", "Graça Morais"),
        ("Q10302288", "Paula Rego"),
        
        # Nordic Artists
        ("Q3950643", "Helene Schjerfbeck"),
        ("Q11861215", "Elin Danielson-Gambogi"),
        ("Q4349683", "Fanny Churberg"),
        ("Q11855166", "Amélie Lundahl"),
        ("Q3926865", "Venny Soldan-Brofeldt"),
        ("Q15514413", "Maria Wiik"),
        ("Q11861216", "Elga Sesemann"),
        ("Q11866537", "Helmi Biese"),
        ("Q11883967", "Sigrid Schauman"),
        ("Q4935845", "Ellen Thesleff"),
        ("Q15514414", "Hanna Rönnberg"),
        ("Q15514415", "Eva Cederström"),
        
        # Eastern European Artists
        ("Q272674", "Katarzyna Kobro"),
        ("Q262031", "Maria Jarema"),
        ("Q269670", "Alina Szapocznikow"),
        ("Q240526", "Erna Rosenstein"),
        ("Q11768018", "Teresa Pągowska"),
        ("Q11768019", "Teresa Tyszkiewicz"),
        ("Q9368837", "Teresa Gierzyńska"),
        ("Q11768020", "Teresa Rudowicz"),
        ("Q16561538", "Krystyna Łada-Studnicka"),
        ("Q11715667", "Natalia LL"),
        ("Q11768021", "Maria Pinińska-Bereś"),
        ("Q11768022", "Izabella Gustowska"),
        
        # Russian Artists
        ("Q182425", "Zinaida Serebriakova"),
        ("Q260080", "Olga Rozanova"),
        ("Q232391", "Varvara Stepanova"),
        ("Q257941", "Aleksandra Ekster"),
        ("Q241638", "Lyubov Popova"),
        ("Q467988", "Vera Mukhina"),
        ("Q242770", "Nadezhda Udaltsova"),
        ("Q4521954", "Elena Guro"),
        ("Q433962", "Kseniya Boguslavskaya"),
        ("Q4088089", "Maria Blanchard"),
        ("Q437792", "Antonina Sofronova"),
        ("Q4091954", "Mariya Bronstein"),
        ("Q4092037", "Maria Vassilieff"),
        ("Q4095382", "Maria Sinyakova"),
        ("Q2628504", "Vera Pestel"),
        
        # Czech Artists
        ("Q10858653", "Zdenka Braunerová"),
        ("Q16017241", "Marie Čermínová"),
        ("Q438433", "Milena Jesenská"),
        ("Q10547574", "Růžena Zátková"),
        ("Q12037562", "Vlasta Vostřebalová Fischerová"),
        ("Q95388057", "Marie Fischerová-Kvěchová"),
        ("Q27914176", "Hana Wichterlová"),
        ("Q12774809", "Eva Švankmajerová"),
        ("Q95161975", "Adriena Šimotová"),
        ("Q2827927", "Běla Kolářová"),
        
        # Hungarian Artists
        ("Q1056720", "Margit Kovács"),
        ("Q847730", "Noémi Ferenczy"),
        ("Q1123896", "Valéria Dénes"),
        ("Q16016813", "Ilka Gedő"),
        ("Q1064804", "Erzsébet Korb"),
        ("Q729849", "Margit Anna"),
        ("Q1056721", "Lili Ország"),
        ("Q1064805", "Erzsi Udvary"),
        ("Q1064806", "Júlia Vajda"),
        
        # Romanian Artists
        ("Q12735480", "Cecilia Cuțescu-Storck"),
        ("Q18539666", "Nina Arbore"),
        ("Q5586808", "Olga Greceanu"),
        ("Q18540623", "Irina Codreanu"),
        ("Q12735481", "Margareta Sterian"),
        ("Q12735482", "Miliţa Petraşcu"),
        ("Q7450116", "Hedda Sterne"),
        
        # Bulgarian Artists
        ("Q12272925", "Elisaveta Konsulova-Vazova"),
        ("Q12278448", "Vessa Petkova"),
        ("Q12283799", "Nevena Stefanova"),
        ("Q12295906", "Tsvetana Alekhina"),
        
        # Greek Artists
        ("Q12875677", "Sophia Laskaridou"),
        ("Q16061908", "Thalia Flora-Karavia"),
        ("Q41312262", "Eleni Zongolopoulos"),
        
        # Turkish Artists
        ("Q6045503", "Fahrelnissa Zeid"),
        ("Q6116870", "Mihri Müşfik Hanım"),
        ("Q6066665", "Hale Asaf"),
        ("Q19517977", "Nezihe Muhiddin"),
        
        # Japanese Artists
        ("Q1133605", "Uemura Shōen"),
        ("Q11572548", "Ono no Komachi"),
        ("Q3181707", "Katsushika Ōi"),
        ("Q11649041", "Yamakawa Shūhō"),
        ("Q11474534", "Okuhara Seiko"),
        ("Q11574741", "Noguchi Shōhin"),
        ("Q11620244", "Yoshida Shūran"),
        ("Q17219102", "Ema Saikō"),
        ("Q11651195", "Yanagawa Seigan"),
        
        # Chinese Artists
        ("Q15917408", "Guan Daosheng"),
        ("Q11303553", "Ma Quan"),
        ("Q5566840", "Chen Shu"),
        ("Q16935050", "Wen Shu"),
        ("Q45428561", "Jin Nong"),
        ("Q10879536", "Cai Wenji"),
        ("Q5092073", "Li Qingzhao"),
        
        # Korean Artists
        ("Q12612428", "Shin Saimdang"),
        ("Q12592065", "Na Hye-sok"),
        ("Q12615743", "Park Re-hyun"),
        ("Q16080474", "Chun Kyung-ja"),
        
        # Indian Artists
        ("Q466906", "Amrita Sher-Gil"),
        ("Q16149263", "Sunayani Devi"),
        ("Q18922025", "Devika Rani"),
        
        # Mexican Artists
        ("Q438434", "María Izquierdo"),
        ("Q3292632", "Aurora Reyes Flores"),
        ("Q5948542", "Lola Álvarez Bravo"),
        ("Q432937", "Nahui Olin"),
        ("Q3306513", "Cordelia Urueta"),
        ("Q5952833", "Elena Huerta"),
        ("Q5666768", "Fanny Rabel"),
        ("Q1243978", "Kati Horna"),
        
        # Brazilian Artists
        ("Q460250", "Anita Malfatti"),
        ("Q442468", "Djanira da Motta e Silva"),
        ("Q10301547", "Fayga Ostrower"),
        ("Q10357642", "Noemia Mourão"),
        ("Q10301548", "Tomie Ohtake"),
        
        # Argentine Artists
        ("Q12165835", "Lola Mora"),
        ("Q441984", "Norah Borges"),
        ("Q450311", "Raquel Forner"),
        ("Q2892778", "Lidy Prati"),
        
        # Chilean Artists
        ("Q4354430", "Rebeca Matte"),
        ("Q3132622", "Henriette Petit"),
        ("Q6143767", "Inés Puyó"),
        
        # Colombian Artists
        ("Q515450", "Débora Arango"),
        ("Q538831", "Emma Reyes"),
        ("Q1814836", "Beatriz González"),
        
        # Cuban Artists
        ("Q441748", "Amelia Peláez"),
        ("Q9029523", "María Magdalena Campos-Pons"),
        
        # Contemporary Artists
        ("Q18559769", "Ghada Amer"),
        ("Q234253", "Mona Hatoum"),
        ("Q231454", "Shirin Neshat"),
        ("Q29579", "Yoko Ono"),
        ("Q15206795", "Lee Bul"),
        ("Q461747", "Sophie Calle"),
        ("Q168647", "Annette Messager"),
        ("Q261669", "Louise Lawler"),
        ("Q265421", "Sherrie Levine"),
        ("Q234768", "Laurie Anderson"),
        ("Q153165", "Rebecca Horn"),
        ("Q238258", "Valie Export"),
        ("Q290245", "Maria Lassnig"),
        ("Q461761", "Eva Hesse"),
        ("Q159063", "Niki de Saint Phalle"),
        ("Q237142", "Joan Jonas"),
        ("Q232152", "Martha Rosler"),
        ("Q273579", "Andrea Fraser"),
        ("Q467547", "Marlene Dumas"),
        ("Q460188", "Yayoi Kusama"),
        ("Q157062", "Carolee Schneemann"),
        ("Q153711", "Judy Chicago"),
        ("Q235434", "Faith Ringgold"),
        ("Q443368", "Barbara Kruger"),
        ("Q158664", "Cindy Sherman"),
        ("Q188507", "Jenny Holzer"),
        ("Q231383", "Kiki Smith"),
        ("Q463597", "Sarah Sze"),
        ("Q235635", "Julie Mehretu"),
        ("Q452025", "Ellen Gallagher"),
        ("Q451553", "Wangechi Mutu"),
        ("Q237132", "Lorna Simpson"),
        ("Q434078", "Carrie Mae Weems"),
        ("Q461459", "Kara Walker"),
        ("Q451340", "Kerry James Marshall"),
        ("Q440847", "Julia Margaret Cameron"),
        ("Q435889", "Gertrude Käsebier"),
        ("Q276319", "Anne Brigman"),
        ("Q435874", "Doris Ulmann"),
        ("Q3075688", "Frances Benjamin Johnston"),
        
        # Lesser-documented European Artists
        ("Q18508668", "Marie-Geneviève Navarre"),
        ("Q3295251", "Marie-Pauline Capelle"),
        ("Q21458567", "Marguerite Jeanne Carpentier"),
        ("Q18508882", "Jeanne Rongier"),
        ("Q3296038", "Marie-Suzanne Roslin"),
        ("Q18002238", "Césarine Henriette Flore Davin-Mirvault"),
        ("Q15971062", "Isabelle Pinson"),
        ("Q18508668", "Marie Bouliard"),
        ("Q3296102", "Joséphine Sarazin de Belmont"),
        
        # Art Nouveau/Decorative Artists
        ("Q265477", "Margaret Macdonald Mackintosh"),
        ("Q3473298", "Frances MacDonald MacNair"),
        ("Q467703", "Jessie M. King"),
        ("Q22915239", "Ann Macbeth"),
        ("Q18761894", "Helen Hay"),
        
        # Miniaturists
        ("Q16065618", "Anne Vallayer-Coster"),
        ("Q18559686", "Henriette de Beaulieu"),
        ("Q18508668", "Marie-Gabrielle Capet"),
        ("Q18559696", "Rosalba Carriera"),
        
        # Folk/Outsider Artists
        ("Q451717", "Grandma Moses"),
        ("Q260683", "Séraphine Louis"),
        ("Q451642", "Aloïse Corbaz"),
        ("Q451643", "Madge Gill"),
        
        # Surrealist Circle
        ("Q262343", "Valentine Hugo"),
        ("Q261223", "Jacqueline Lamba"),
        ("Q3306445", "Alice Rahon"),
        ("Q255995", "Stella Snead"),
        
        # Bloomsbury Group Adjacent
        ("Q2338009", "Dorothy Brett"),
        ("Q1384160", "Dora Carrington"),
        ("Q16749407", "Barbara Bagenal"),
        
        # Latin American Muralists
        ("Q438428", "Aurora Reyes"),
        ("Q9015543", "Rina Lazo"),
        ("Q7316352", "Fanny Rabel"),
        
        # Australian Impressionists
        ("Q16065618", "Grace Cossington Smith"),
        ("Q6779414", "Clarice Beckett"),
        ("Q16057954", "Vida Lahey"),
        
        # New Zealand Artists
        ("Q241936", "Frances Hodgkins"),
        ("Q449674", "Rita Angus"),
        ("Q16015045", "Olivia Spencer Bower"),
        
        # South African Artists
        ("Q17416642", "Irma Stern"),
        ("Q441374", "Maggie Laubser"),
        ("Q16728686", "Bertha Everard"),
        
        # Canadian Group of Seven Era
        ("Q17525670", "Prudence Heward"),
        ("Q7356352", "Lilias Torrance Newton"),
        ("Q7245456", "Pegi Nicol MacLeod"),
        
        # Abstract Expressionist Adjacent
        ("Q441555", "Perle Fine"),
        ("Q16002673", "Sonia Gechtoff"),
        ("Q7562080", "Ethel Schwabacher"),
        
        # Bay Area Figurative
        ("Q538012", "Joan Brown"),
        ("Q16013282", "Jay DeFeo"),
        
        # Photorealist Movement
        ("Q461766", "Audrey Flack"),
        ("Q15493451", "Carolyn Brady"),
        
        # Pattern and Decoration Movement
        ("Q441480", "Valerie Jaudon"),
        ("Q16195109", "Howardena Pindell"),
        
        # Feminist Art Movement
        ("Q16002769", "Mary Beth Edelson"),
        ("Q539654", "Suzanne Lacy"),
        
        # Young British Artists
        ("Q449856", "Gillian Wearing"),
        ("Q449409", "Sam Taylor-Johnson"),
        
        # Street Artists
        ("Q16956490", "Lady Pink"),
        ("Q441928", "Swoon"),
        ("Q16196393", "Maya Hayuk"),
        
        # Digital/New Media
        ("Q450661", "Lillian Schwartz"),
        ("Q434077", "Vera Molnár"),
        ("Q21394161", "Sarah Anne Drake"),
        ("Q4821458", "Augusta Innes Withers"),
        ("Q123280", "Berthe Hoola van Nooten"),
        
        # Medieval Illuminators
        ("Q257351", "Herrad of Landsberg"),
        ("Q26876803", "Ende"),
        ("Q88955", "Claricia"),
        ("Q27947668", "Diemudis"),
        
        # Court Painters
        ("Q430136", "Levina Teerlinc"),
        ("Q265828", "Anna Dorothea Therbusch"),
        ("Q468741", "Anne Vallayer-Coster"),
        
        # Italian Renaissance/Baroque
        ("Q271475", "Elisabetta Sirani"),
        ("Q281033", "Lavinia Fontana"),
        
        # Royal Academy Members
        ("Q233351", "Mary Moser"),
        
        # Chinese Historical Painters
        ("Q265237", "Guan Daosheng"),
        ("Q823543", "Chen Shu"),
        
        # Australian Artists
        ("Q7322502", "Rica Erickson"),
        
        # British Botanical Artists
        ("Q18922192", "Frances Elizabeth Tripp"),
        ("Q522035", "Mary Elizabeth Barber"),
        
        # American Regional Artists
        ("Q4883479", "Belle Baranceanu"),
        ("Q20856506", "Gene Kloss"),
        
        # Miniaturists
        ("Q18917950", "Susannah-Penelope Rosse"),
        ("Q56679263", "Mary Roberts"),
        
        # Dutch Golden Age
        ("Q1662640", "Alida Withoos"),
        
        # Mexican Artists
        ("Q438434", "María Izquierdo"), # You may already have this one
        
        # Additional well-documented artists from the research
        ("Q265222", "Anna Ancher"), # Danish Skagen painter
        ("Q444223", "Marianne North"), # British botanical painter
        ("Q233207", "Leonora Carrington"), # British-Mexican surrealist
        ("Q235281", "Helen Frankenthaler"), # Abstract Expressionist
        ("Q469934", "Joan Mitchell"), # Abstract Expressionist
        ("Q237959", "Lee Krasner"), # Abstract Expressionist
        ("Q156773", "Marie Bashkirtseff"), 
        ("Q7150434", "Pablita Velarde"),
        ("Q3482664", "Maria Martinez"),
        ("Q7971505", "Tonita Peña"),
        
        # Folk/Naïve Artists
        ("Q273880", "Séraphine Louis"),
        ("Q290253", "Aloïse Corbaz"),
        
        # Australian Indigenous Artists
        ("Q4895246", "Emily Kame Kngwarreye"),
        ("Q16013642", "Kathleen Petyarre"),
        ("Q4768723", "Minnie Pwerle"),
        
        # New Zealand Artists  
        ("Q6138350", "Evelyn Page"),
        ("Q21456888", "May Smith"),
        
        # More Scandinavian Artists
        ("Q259206", "Harriet Backer"),
        ("Q4935232", "Asta Nørregaard"),
        
        # Pacific/Hawaiian Artists
        ("Q21995141", "Madge Tennent"),
        
        # More Latin American Artists
        ("Q5665841", "Olga de Amaral"),
        ("Q2881954", "Beatriz Milhazes"),
        
        # African American Folk Artists
        ("Q15060026", "Clementine Hunter"),
        ("Q451642", "Horace Pippin"),
        
        # More British Artists
        ("Q16065633", "Annie Swynnerton"),
        ("Q19402734", "Ethel Walker"),
        # Botanical/Scientific Illustrators
        ("Q18761840", "Margaret Mee"),
        ("Q1702368", "Marianne North"),
        ("Q3180537", "Elizabeth Blackwell"),
        ("Q447678", "Anne Pratt"),
        ("Q2870951", "Augusta Innes Withers"),
        
        # Australian Artists
        ("Q4768670", "Margaret Olley"),
        ("Q7928936", "Violet Teague"),
        ("Q21456571", "Dora Meeson"),
        ("Q2962070", "Stella Bowen"),
        ("Q5387041", "Portia Geach"),
        
        # South African Artists
        ("Q20675334", "Bertha Everard"),
        ("Q18636488", "Dorothy Kay"),
        ("Q5260681", "Penny Siopis"),
        
        # Indian/South Asian Artists
        ("Q18419523", "Sita Devi"),
        ("Q13104371", "Arpita Singh"),
        ("Q4806547", "Nalini Malani"),
        
        # Japanese Artists (beyond those listed)
        ("Q11613479", "Kiyohara Yukinobu"),
        ("Q11570999", "Nakabayashi Seishuku"),
        ("Q11652984", "Yoshida Hiroshi"),
        
        # Korean Artists (additional)
        ("Q16080693", "Kim Yun-sin"),
        ("Q12591783", "Lee Sook-ja"),
        
        # Southeast Asian Artists
        ("Q19518994", "Georgette Chen"),
        ("Q21706515", "Amanda Heng"),
        
        # Middle Eastern Artists
        ("Q4116742", "Inji Aflatoun"),
        ("Q2919969", "Monir Shahroudy Farmanfarmaian"),
        
        # Soviet/Russian Artists (additional)
        ("Q4407860", "Anna Ostroumova-Lebedeva"),
        ("Q4100861", "Maria Yakunchikova"),
        ("Q4173054", "Zinaida Volkonskaya"),
        
        # Polish Artists (additional)
        ("Q9175930", "Maria Dulębianka"),
        ("Q11771803", "Zofia Albinowska-Minkiewiczowa"),
        
        # Czech/Slovak Artists
        ("Q10858653", "Zdenka Braunerová"),
        ("Q12037562", "Vlasta Vostřebalová"),
        
        # Hungarian Artists (additional)
        ("Q897988", "Mária Szántó"),
        ("Q1149881", "Róza Lévai"),
        
        # Yugoslav/Balkan Artists
        ("Q16111642", "Nadežda Petrović"),
        ("Q437995", "Milena Pavlović-Barili"),
        
        # Danish Artists (additional)
        ("Q13640", "Anna Petersen"),
        ("Q4775373", "Bertha Wegmann"),
        
        # Norwegian Artists
        ("Q11989289", "Kitty Kielland"),
        ("Q6068086", "Harriet Backer"),
        
        # Finnish Artists (additional)
        ("Q4349683", "Fanny Churberg"),
        ("Q11879384", "Elin Alfhild Nordlund"),
        
        # Dutch 20th Century
        ("Q5598822", "Co Westerik"),
        ("Q2740775", "Bep Rietveld"),
        
        # Belgian Additional
        ("Q21338577", "Berthe Dubail"),
        ("Q16674035", "Marie-Jo Lafontaine"),
        
        # Early American Artists
        ("Q20856817", "Ellen Day Hale"),
        ("Q5384049", "Cornelia Adele Strong Fassett"),
        
        # American Folk Artists
        ("Q460571", "Grandma Moses"),
        ("Q7733129", "Sister Gertrude Morgan"),
        
        # American Modernists
        ("Q5280556", "Florine Stettheimer"),
        ("Q5365317", "Marguerite Zorach"),
        
        # Canadian Artists (additional)
        ("Q19665155", "Paraskeva Clark"),
        ("Q7180926", "Pegi Nicol MacLeod"),
        
        # Mexican Artists (additional)
        ("Q6013503", "Lola Cueto"),
        ("Q450659", "Alice Rahon"),
        
        # Colombian Artists
        ("Q5943469", "Lucy Tejada"),
        ("Q5941677", "Ana Mercedes Hoyos"),
        
        # Venezuelan Artists
        ("Q9011816", "Elsa Gramcko"),
        ("Q5879813", "Gego"),
        
        # Uruguayan Artists
        ("Q12165835", "María Freire"),
        ("Q2877033", "Amalia Nieto"),
        
        # Persian/Iranian Artists
        ("Q5933163", "Mansooreh Hosseini"),
        ("Q16733520", "Behjat Sadr"),
        
        # Egyptian Artists
        ("Q12209017", "Gazbia Sirry"),
        ("Q4691678", "Tahia Halim"),
        
        # Lebanese Artists
        ("Q6120893", "Saloua Raouda Choucair"),
        ("Q19958935", "Huguette Caland"),
        
        # More Miniaturists
        ("Q94748244", "Catherine da Costa"),
        ("Q18508916", "Joan Carlile"),
        
        # Scottish Artists
        ("Q19974733", "Anne Forbes"),
        ("Q18530282", "Katherine Cameron"),
        
        # Welsh Artists
        ("Q13127679", "Gwen John"),
        ("Q18535661", "Margaret Lindsay Williams"),
        
        # Irish Artists (additional)
        ("Q7377012", "Rosamond Jacob"),
        ("Q15429326", "Beatrice Elvery"),
        
        # More Contemporary Artists
        ("Q5384049", "Miriam Schapiro"),
        ("Q5337781", "Harmony Hammond"),
        ("Q7615486", "Emma Amos"),
        ("Q16012438", "Faith Wilding"),
        ("Q5403935", "Nancy Fried"),
        
        # Digital/New Media Artists
        ("Q11626202", "Vera Molnar"),
        ("Q537810", "Lynn Hershman Leeson"),
        
        # Performance/Conceptual Artists who also paint
        ("Q7916517", "Janine Antoni"),
        ("Q6148087", "Sophie Calle"),
        
        # Missing Historical Artists
        ("Q2518618", "Gesina ter Borch"),
        ("Q19507673", "Mayken Verhulst"),
        ("Q20978051", "Diana Mantuana"),
        ("Q18684864", "Lucrina Fetti"),
        ("Q3950627", "Isabella Piccini"),
        
        # Court Artists
        ("Q18115787", "Magdalena de Passe"),
        ("Q94647838", "Giovanna Fratellini")
    ]
    


    # Try to get more from Wikidata
    query = """
    SELECT DISTINCT ?artist ?artistLabel WHERE {
      ?artist wdt:P21 wd:Q6581072 .  # female
      ?artist wdt:P106 wd:Q1028181 .  # painter
      ?painting wdt:P31 wd:Q3305213 . # has paintings
      ?painting wdt:P170 ?artist .
      
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en,fr,it,de,es" .
      }
    }
    LIMIT 100
    """
    
    results = execute_sparql_query(query, "additional female artists")
    
    # Combine known and queried
    all_artists = list(known_female)
    seen_qids = {a[0] for a in all_artists}
    
    for r in results:
        qid = r.get('artist', '').split('/')[-1]
        name = r.get('artistLabel', '')
        if qid and name and qid not in seen_qids and name != qid:
            all_artists.append((qid, name))
            seen_qids.add(qid)
    
    return all_artists


# === EXTRA FEMALE ARTISTS FETCHER =============================================
# Finds 500 *new* female artists (human, female) with at least one painting on Wikidata
# Excludes all QIDs already present in your `known_female` list.

import time, csv, json, requests
from typing import Set, Tuple, List

WDQS = "https://query.wikidata.org/sparql"
UA   = "GentileschiDB/1.0 (mailto:you@example.com)"  # <- put your email or project URL

SPARQL_TEMPLATE = """
SELECT ?artist ?artistLabel (COUNT(?w) AS ?count)
WHERE {
  ?w wdt:P31 wd:Q3305213 ;   # painting
     wdt:P170 ?artist .      # creator
  ?artist wdt:P21 wd:Q6581072 .  # female
  ?artist wdt:P31 wd:Q5 .        # human
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?artist ?artistLabel
ORDER BY DESC(?count)
LIMIT {limit} OFFSET {offset}
"""

def _run_wdqs(query: str) -> List[Tuple[str, str, int]]:
    r = requests.get(
        WDQS,
        params={"format": "json", "query": query},
        headers={"User-Agent": UA},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    out = []
    for b in data.get("results", {}).get("bindings", []):
        qid = b["artist"]["value"].rpartition("/")[-1]
        name = b.get("artistLabel", {}).get("value", "")
        try:
            cnt = int(b.get("count", {}).get("value", "1"))
        except:
            cnt = 1
        out.append((qid, name, cnt))
    return out

def fetch_extra_female_artists(target: int = 500, page_size: int = 500, max_pages: int = 50,
                               sleep_sec: float = 1.0) -> List[Tuple[str, str]]:
    """
    Returns up to `target` (qid, name) pairs NOT in your existing `known_female` list.
    Writes CSV: female_artists_extra_500.csv with columns: qid, name, painting_count.
    """
    try:
        existing_qids: Set[str] = {qid for (qid, _name) in known_female}
    except NameError:
        existing_qids = set()

    results: List[Tuple[str, str, int]] = []
    collected_qids: Set[str] = set()
    offset = 0

    for _page in range(max_pages):
        q = SPARQL_TEMPLATE.format(limit=page_size, offset=offset)
        batch = _run_wdqs(q)
        if not batch:
            break

        # Filter out already-known artists and duplicates
        for qid, name, cnt in batch:
            if qid in existing_qids or qid in collected_qids:
                continue
            results.append((qid, name, cnt))
            collected_qids.add(qid)
            if len(collected_qids) >= target:
                break

        if len(collected_qids) >= target:
            break

        offset += page_size
        time.sleep(sleep_sec)  # be polite to WDQS

    # Trim exactly to `target`
    results = results[:target]

    # Save CSV
    with open("female_artists_extra_500.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "name", "painting_count"])
        for qid, name, cnt in results:
            w.writerow([qid, name, cnt])

    # Also print a Python literal 
    tuple_list = [(f"Q{qid}", name) for qid, name, _cnt in results]
    print(f"\n✓ Saved female_artists_extra_500.csv with {len(results)} rows.")
    print("Pasteable Python literal (first few):")
    print(json.dumps(tuple_list[:10], ensure_ascii=False, indent=2))
    return [(f"Q{qid}", name) for qid, name, _ in results]


def get_male_artists_simple() -> List[Tuple[str, str]]:
    """Get male artists - simplified query"""
    # Start with known male artists
    known_male = [
        ("Q5593", "Pablo Picasso"),
        ("Q5599", "Rembrandt"),
        ("Q762", "Leonardo da Vinci"),
        ("Q5589", "Vincent van Gogh"),
        ("Q5582", "Claude Monet"),
        ("Q5597", "Diego Velázquez"),
        ("Q296", "Johannes Vermeer"),
        ("Q104326", "Salvador Dalí"),
        ("Q7624", "Caravaggio"),
        ("Q5577", "Peter Paul Rubens"),
        ("Q46408", "Jackson Pollock"),
        ("Q41264", "Edgar Degas"),
        ("Q33231", "Édouard Manet"),
        ("Q40599", "Henri de Toulouse-Lautrec"),
        ("Q5598", "Jan van Eyck"),
        ("Q131767", "Francisco Goya"),
        ("Q185030", "Wassily Kandinsky"),
        ("Q44007", "Paul Klee"),
        ("Q42207", "Andy Warhol"),
        ("Q164765", "Jean-Michel Basquiat"),
        ("Q35548", "Paul Cézanne"),
        ("Q37693", "Paul Gauguin"),
        ("Q156391", "Georges Braque"),
        ("Q5592", "Michelangelo"),
        ("Q5603", "Raphael"),
        ("Q160538", "David Hockney"),
        ("Q47551", "Francis Bacon"),
        ("Q151679", "René Magritte"),
        ("Q157666", "Mark Rothko"),
        ("Q37001", "Lucian Freud"),
        ("Q5580", "Sandro Botticelli"),
        ("Q7751", "Pierre-Auguste Renoir"),
        ("Q130777", "Gustav Klimt"),
        ("Q44356", "Egon Schiele"),
        ("Q5600", "Albrecht Dürer"),
        ("Q164349", "Titian"),
        ("Q191748", "Hieronymus Bosch"),
        ("Q9440", "Pieter Bruegel the Elder"),
        ("Q41421", "Georges Seurat"),
        ("Q161170", "Henri Matisse"),
        ("Q153774", "Amedeo Modigliani"),
        ("Q154585", "Edward Hopper"),
        ("Q137183", "Marc Chagall"),
        ("Q152542", "Piet Mondrian"),
        ("Q239007", "Joan Miró"),
        ("Q102272", "J. M. W. Turner"),
        ("Q159297", "John Constable"),
        ("Q130650", "Eugène Delacroix"),
        ("Q41554", "Caspar David Friedrich"),
        ("Q34661", "Gustav Courbet"),
    ]
    
    # Try to get more from Wikidata
    query = """
    SELECT DISTINCT ?artist ?artistLabel WHERE {
      ?artist wdt:P21 wd:Q6581097 .  # male
      ?artist wdt:P106 wd:Q1028181 .  # painter
      ?painting wdt:P31 wd:Q3305213 . # has paintings
      ?painting wdt:P170 ?artist .
      
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en,fr,it,de,es" .
      }
    }
    LIMIT 100
    """
    
    results = execute_sparql_query(query, "additional male artists")
    
    # Combine
    all_artists = list(known_male)
    seen_qids = {a[0] for a in all_artists}
    
    for r in results:
        qid = r.get('artist', '').split('/')[-1]
        name = r.get('artistLabel', '')
        if qid and name and qid not in seen_qids and name != qid:
            all_artists.append((qid, name))
            seen_qids.add(qid)
    
    return all_artists

def get_paintings_for_artist(artist_qid: str, artist_name: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Get paintings for a specific artist"""
    query = f"""
    SELECT DISTINCT ?painting ?paintingLabel ?image ?inception
           ?collection ?collectionLabel ?location ?locationLabel
           ?inventory ?genre ?genreLabel ?material ?materialLabel
           ?height ?width
    WHERE {{
      ?painting wdt:P31 wd:Q3305213 .     # painting
      ?painting wdt:P170 wd:{artist_qid} . # by this artist
      ?painting wdt:P18 ?image .           # has image
      
      OPTIONAL {{ ?painting wdt:P571 ?inception }}
      OPTIONAL {{ ?painting wdt:P195 ?collection }}
      OPTIONAL {{ ?painting wdt:P276 ?location }}
      OPTIONAL {{ ?painting wdt:P217 ?inventory }}
      OPTIONAL {{ ?painting wdt:P136 ?genre }}
      OPTIONAL {{ ?painting wdt:P186 ?material }}
      OPTIONAL {{ ?painting wdt:P2048 ?height }}
      OPTIONAL {{ ?painting wdt:P2049 ?width }}
      
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en,fr,it,de,es" .
      }}
    }}
    LIMIT {limit}
    """
    
    return execute_sparql_query(query, f"paintings by {artist_name}")

from collections import defaultdict, deque
import random

def artist_round_robin_sample(rows: List[Dict[str, Any]], target_n: int) -> List[Dict[str, Any]]:
    """Diverse sub-sampling by artist: round-robin across artists until target_n."""
    if target_n >= len(rows):
        return list(rows)
    by_artist = defaultdict(list)
    for r in rows:
        by_artist[r.get('artist_name','Unknown')].append(r)
    # shuffle each artist's list and the artist order for variety
    queues = []
    for k,v in by_artist.items():
        random.shuffle(v)
        queues.append(deque(v))
    random.shuffle(queues)
    out = []
    i = 0
    while len(out) < target_n and queues:
        q = queues[i % len(queues)]
        if q:
            out.append(q.popleft())
            i += 1
        else:
            queues.pop(i % len(queues))
            if queues:
                i = i % len(queues)
    return out


# --- Commons Functions ---
def commons_title_from_url(url: str) -> Optional[str]:
    """Extract Commons file title from image URL"""
    if not url:
        return None
    
    if "Special:FilePath/" in url:
        part = url.split("Special:FilePath/", 1)[-1].split("?", 1)[0]
        return "File:" + unquote(part)
    elif "/File:" in url:
        return "File:" + url.split("/File:", 1)[-1]
    elif "/thumb/" in url:
        # Extract from thumb URL
        parts = url.split('/')
        for i, part in enumerate(parts):
            if part == 'thumb' and i + 3 < len(parts):
                return "File:" + unquote(parts[i + 3])
    
    return None

def commons_imageinfo(title: str, width: int = TARGET_LONG_EDGE) -> Optional[Dict[str, Any]]:
    """Get Commons image info with licensing"""
    params = {
        "action": "query",
        "prop": "imageinfo",
        "titles": title,
        "iiprop": "url|size|mime|sha1|extmetadata",
        "iiurlwidth": str(width),
        "format": "json",
        "formatversion": "2",
        "uselang": "en",
    }
    
    try:
        r = http_get(COMMONS_API, params=params)
        data = r.json()
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        
        pg = pages[0]
        if "missing" in pg:
            return None
        
        infos = pg.get("imageinfo", [])
        if not infos:
            return None
        
        ii = infos[0]
        em = ii.get("extmetadata", {}) or {}
        
        def emv(k):
            v = em.get(k, {})
            return v.get("value") if isinstance(v, dict) else v
        
        return {
            "title": title,
            "width": ii.get("width"),
            "height": ii.get("height"),
            "mime": ii.get("mime"),
            "sha1": ii.get("sha1"),
            "original_url": ii.get("url"),
            "thumb_url": ii.get("thumburl") or ii.get("url"),
            "thumb_width": ii.get("thumbwidth") or ii.get("width"),
            "thumb_height": ii.get("thumbheight") or ii.get("height"),
            "license_short": emv("LicenseShortName"),
            "license_url": emv("LicenseUrl"),
            "artist": emv("Artist"),
            "credit": emv("Credit"),
            "attribution": emv("Attribution"),
            "usage_terms": emv("UsageTerms"),
        }
    except Exception as e:
        log.warning(f"Commons API error: {e}")
        return None

def download_url_to_file(url: str, out_path: pathlib.Path) -> bool:
    """Download image file"""
    last_exc = None
    for attempt in range(1, RETRIES+1):
        try:
            r = session.get(url, timeout=90, stream=True)
            r.raise_for_status()
            content = r.content
            if len(content) < MIN_ACCEPT_BYTES:
                raise IOError(f"Content too small: {len(content)} bytes")
            out_path.write_bytes(content)
            return True
        except Exception as e:
            last_exc = e
            time.sleep(RETRY_SLEEP_SEC * attempt)
    log.warning(f"   × Download failed: {last_exc}")
    return False

# --- Collection Functions with Resume ---
def collect_paintings_by_gender(gender: str, target: int = 400, resume: bool = False,
                                max_per_artist: Optional[int] = None) -> List[Dict[str, Any]]:
    """Collect paintings for a gender with proper deduplication and resume capability.
    
    Changes:
    - accepts `max_per_artist` (gender-specific caps supported).
    - uses per-artist cap consistently when fetching and appending.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Collecting {gender.upper()} paintings (target: {target})")
    log.info(f"{'='*60}")
    
    # Load existing data if resuming
    existing_rows, seen_qids, artist_counts = [], set(), {}
    if resume:
        existing_rows, seen_qids, artist_counts = load_existing_data(gender)
        if existing_rows:
            log.info(f"Resuming: {len(existing_rows)} paintings already collected")
            if len(existing_rows) >= target:
                log.info(f"Already have {len(existing_rows)} paintings (>= target of {target})")
                return existing_rows
    
    # Determine per-artist cap
    if max_per_artist is None:
        try:
            max_per_artist = MAX_PER_ARTIST_FEMALE if gender == "female" else MAX_PER_ARTIST_MALE
        except NameError:
            max_per_artist = MAX_PER_ARTIST  # fallback to legacy cap
    
    # Get artist list
    artists = get_female_artists_simple() if gender == "female" else get_male_artists_simple()
    log.info(f"Found {len(artists)} {gender} artists")
    
    all_paintings = list(existing_rows)  # Start with existing data
    
    # Shuffle for variety
    import random
    random.shuffle(artists)
    
    for artist_qid, artist_name in artists:
        if len(all_paintings) >= target:
            break
        
        # Respect per-artist cap if resuming
        if artist_counts.get(artist_name, 0) >= max_per_artist:
            continue
        
        # Fetch up to the cap for this artist
        paintings_data = get_paintings_for_artist(artist_qid, artist_name, limit=max_per_artist)
        if not paintings_data:
            continue
        
        added = 0
        for p in paintings_data:
            painting_qid = (p.get('painting') or '').split('/')[-1]
            if not painting_qid:
                continue
            
            # Skip duplicates
            if painting_qid in seen_qids:
                continue
            
            # Skip if we already reached the cap for this artist
            current_count = artist_counts.get(artist_name, 0)
            if current_count >= max_per_artist:
                break
            
            # Build painting record (keep keys used downstream; default empties for others)
            painting = {
                'qid': painting_qid,
                'label': p.get('paintingLabel', 'Untitled'),
                'description': '',
                'image_url': p.get('image', ''),
                'artist_name': artist_name,
                'artist_qid': artist_qid,
                'artist_gender': gender,
                'birth_year': '',
                'death_year': '',
                'nationality': '',
                'inventory': p.get('inventory', ''),
                'collection': p.get('collectionLabel', ''),
                'location': p.get('locationLabel', ''),
                'inception': extract_year(p.get('inception', '')),
                'genre': p.get('genreLabel', ''),
                'material': p.get('materialLabel', ''),
                'height_cm': p.get('height', ''),
                'width_cm': p.get('width', ''),
                'commons_title': '',
                'commons_url': '',
                'commons_license': '',
                'commons_license_url': '',
                'commons_attribution': '',
                'commons_credit': '',
                'commons_usage_terms': '',
                'image_width': '',
                'image_height': '',
                'image_mime': '',
                'image_sha1': '',
                'local_filename': '',
                'image_local_path': ''
            }
            
            all_paintings.append(painting)
            seen_qids.add(painting_qid)
            artist_counts[artist_name] = current_count + 1
            added += 1
            
            if len(all_paintings) >= target:
                break
        
        if added > 0:
            log.info(f"    Added {added} paintings (total: {len(all_paintings)})")
        
        time.sleep(1)  # Rate limiting
    
    log.info(f"\nCollected {len(all_paintings)} unique {gender} paintings from {len(artist_counts)} artists")
    return all_paintings

# --- Enrich and Download with Resume ---
def enrich_and_download(records: List[Dict[str, Any]], out_dir: str, gender: str, resume: bool = False) -> List[Dict[str, Any]]:
    """Download images and enrich with Commons metadata"""
    ensure_dir(out_dir)
    out = []
    
    # Get already downloaded files if resuming
    downloaded_files = get_downloaded_files(out_dir) if resume else set()
    
    for i, rec in enumerate(records, 1):
        title = rec["label"]
        year = rec.get("inception") or "undated"
        
        # Generate filename
        safe_title = slugify(title)
        safe_artist = slugify(rec['artist_name'])
        prefix = "F" if gender == "female" else "M"
        filename = f"{prefix}_{i:03d}_{safe_artist}_{safe_title}_{year}.jpg"
        final_path = pathlib.Path(out_dir) / filename
        
        # Skip if already downloaded and resuming
        if resume and rec.get('local_filename'):
            # Already has local filename in CSV
            out.append(rec)
            log.info(f"[{gender[0].upper()}_{i:03d}] {title}\n   ✓ Already processed")
            continue
        elif resume and filename in downloaded_files:
            # File exists but not in CSV yet
            rec['local_filename'] = filename
            rec['image_local_path'] = str(final_path)
            out.append(rec)
            log.info(f"[{gender[0].upper()}_{i:03d}] {title}\n   ✓ Already downloaded")
            continue
        
        if not rec.get('image_url'):
            log.info(f"[{gender[0].upper()}_{i:03d}] {title}\n   × No image URL. Skipping.")
            continue
        
        log.info(f"[{gender[0].upper()}_{i:03d}] {title}")
        
        # Get Commons metadata
        commons_title = commons_title_from_url(rec['image_url'])
        if commons_title:
            rec['commons_file_title'] = commons_title
            info = commons_imageinfo(commons_title, width=TARGET_LONG_EDGE)
            if info:
                rec['commons_artist'] = info.get('artist', '')
                rec['commons_license'] = info.get('license_short', '')
                rec['commons_license_url'] = info.get('license_url', '')
                rec['commons_attribution'] = info.get('attribution', '')
                rec['commons_credit'] = info.get('credit', '')
                rec['commons_usage_terms'] = info.get('usage_terms', '')
                rec['image_width'] = info.get('thumb_width') or info.get('width', '')
                rec['image_height'] = info.get('thumb_height') or info.get('height', '')
                rec['image_mime'] = info.get('mime', '')
                rec['image_sha1'] = info.get('sha1', '')
                
                # Use thumb URL if available
                url = info.get('thumb_url') or info.get('original_url')
            else:
                url = rec['image_url']
        else:
            url = rec['image_url']
        
        # Download
        ok = download_url_to_file(url, final_path)
        if not ok and rec.get('image_url') != url:
            # Try original if thumb failed
            ok = download_url_to_file(rec['image_url'], final_path)
        
        if ok:
            rec['local_filename'] = filename
            rec['image_local_path'] = str(final_path)
            out.append(rec)
        else:
            log.info("   × Download failed. Skipping.")
        
        time.sleep(0.3)
    
    return out

# --- Write CSV/JSON ---
CSV_FIELDS = [
    'qid', 'label', 'description', 'image_url', 'artist_name', 'artist_qid',
    'artist_gender', 'birth_year', 'death_year', 'nationality', 'inventory', 
    'collection', 'location', 'inception', 'genre', 'depicts', 'material', 
    'height_cm', 'width_cm', 'local_filename', 'image_local_path', 'commons_title', 'commons_url',
    'commons_file_title', 'commons_artist', 'commons_license', 
    'commons_license_url', 'commons_attribution', 'commons_credit',
    'commons_usage_terms', 'image_width', 'image_height', 'image_mime',
    'image_sha1'
]

# --- Write CSV/JSON ---
CSV_FIELDS = [
    'qid', 'label', 'description', 'image_url', 'artist_name', 'artist_qid',
    'artist_gender', 'birth_year', 'death_year', 'nationality', 'inventory',
    'collection', 'location', 'inception', 'genre', 'depicts', 'material',
    'height_cm', 'width_cm', 'local_filename', 'image_local_path',
    'commons_file_title', 'commons_artist', 'commons_license',
    'commons_license_url', 'commons_attribution', 'commons_credit',
    'commons_usage_terms', 'image_width', 'image_height', 'image_mime',
    'image_sha1'
]

def _normalize_csv_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Harmonize row keys so they match CSV_FIELDS and avoid DictWriter errors.
    - Map 'commons_title' -> 'commons_file_title'
    - Prefer 'image_url'; if only 'commons_url' exists, copy into 'image_url'
    - Ensure every CSV field exists (fill with '')
    - Return a filtered dict with only CSV_FIELDS
    """
    row = dict(r)  # shallow copy

    # Key synonyms from earlier revisions
    if 'commons_title' in row and 'commons_file_title' not in row:
        row['commons_file_title'] = row.pop('commons_title')

    if ('image_url' not in row or not row['image_url']) and 'commons_url' in row:
        row['image_url'] = row.get('commons_url', '')

    # Ensure all expected fields exist
    for k in CSV_FIELDS:
        row.setdefault(k, '')

    # Emit only the columns we declare
    return {k: row.get(k, '') for k in CSV_FIELDS}

def write_csv_json(rows: List[Dict[str, Any]], gender: str) -> Tuple[str, str]:
    """Write CSV and JSON files with safe handling of stray keys."""
    csv_file = f"{gender.lower()}_paintings_dataset.csv"
    json_file = f"{gender.lower()}_paintings_dataset.json"

    # CSV: normalize and ignore any remaining extra keys defensively
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(_normalize_csv_row(r))

    # JSON can keep the full original dicts (extra keys are fine in JSON)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    return csv_file, json_file

# --- Statistics ---
def compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics"""
    total = len(rows)
    museums = {}
    artists = {}
    years = []
    centuries = {}
    
    for r in rows:
        # Museums
        m = r.get("collection") or r.get("location") or "Unknown"
        museums[m] = museums.get(m, 0) + 1
        
        # Artists
        a = r.get("artist_name", "Unknown")
        artists[a] = artists.get(a, 0) + 1
        
        # Years
        y = r.get("inception")
        if y:
            try:
                year = int(y)
                years.append(year)
                century = f"{(year // 100) + 1}th century"
                centuries[century] = centuries.get(century, 0) + 1
            except:
                pass
    
    years_sorted = sorted(years) if years else []
    
    def median(lst):
        if not lst: return None
        n = len(lst)
        mid = n//2
        return (lst[mid] if n%2==1 else (lst[mid-1]+lst[mid])/2)
    
    return {
        "count": total,
        "unique_museums": len(museums),
        "unique_artists": len(artists),
        "top_museums": sorted(museums.items(), key=lambda x: (-x[1], x[0]))[:10],
        "top_artists": sorted(artists.items(), key=lambda x: (-x[1], x[0]))[:15],
        "centuries": sorted(centuries.items(), key=lambda x: x[0]),
        "year_min": years_sorted[0] if years_sorted else None,
        "year_max": years_sorted[-1] if years_sorted else None,
        "year_median": median(years_sorted),
    }

# --- HTML Generation ---
def html_badge(text: str, color: str) -> str:
    color_map = {
        "#ff69b4": "#8b2c5e",  # deep magenta for female
        "#4a90e2": "#1e4788",  # deep blue for male
    }
    badge_color = color_map.get(color, color)
    return f'<span style="display:inline-block;padding:3px 10px;border:1px solid {badge_color};font-size:10px;letter-spacing:0.08em;background:transparent;color:{badge_color};font-weight:400;text-transform:uppercase;font-family:\'Courier New\',monospace;">{html.escape(text)}</span>'

def render_stats_block(title: str, stats: Dict[str, Any]) -> str:
    """Render statistics block with centered academic style"""
    
    # Generate only the first 5 rows for top artists and museums
    top_museums = stats.get("top_museums", [])[:5]
    top_artists = stats.get("top_artists", [])[:5]
    centuries = stats.get("centuries", [])
    
    tm_rows = "".join([f"<tr><td style='padding:12px 24px;border-bottom:1px solid #e5dfd6;text-align:left;'>{html.escape(str(k))}</td><td style='padding:12px 24px;border-bottom:1px solid #e5dfd6;text-align:center;font-weight:300;'>{v}</td></tr>" for k,v in top_museums])
    ta_rows = "".join([f"<tr><td style='padding:12px 24px;border-bottom:1px solid #e5dfd6;text-align:left;'>{html.escape(str(k))}</td><td style='padding:12px 24px;border-bottom:1px solid #e5dfd6;text-align:center;font-weight:300;'>{v}</td></tr>" for k,v in top_artists])
    tc_rows = "".join([f"<tr><td style='padding:12px 24px;border-bottom:1px solid #e5dfd6;text-align:left;'>{html.escape(str(k))}</td><td style='padding:12px 24px;border-bottom:1px solid #e5dfd6;text-align:center;font-weight:300;'>{v}</td></tr>" for k,v in centuries])
    
    return f"""
    <section style="margin:64px auto;max-width:1100px;text-align:center;">
      <h2 style="font-size:11px;letter-spacing:0.3em;color:#8b7968;font-weight:400;margin-bottom:32px;text-transform:uppercase;font-family:'Courier New',monospace;">{html.escape(title)}</h2>
      
      <div style="display:flex;justify-content:center;gap:60px;margin-bottom:48px;">
        <div>
          <div style="font-size:13px;color:#8b7968;margin-bottom:8px;letter-spacing:0.05em;">Total Paintings</div>
          <div style="font-size:42px;color:#2c1810;font-weight:300;line-height:1;">{stats.get("count",0)}</div>
        </div>
        <div>
          <div style="font-size:13px;color:#8b7968;margin-bottom:8px;letter-spacing:0.05em;">Unique Artists</div>
          <div style="font-size:42px;color:#2c1810;font-weight:300;line-height:1;">{stats.get("unique_artists",0)}</div>
        </div>
        <div>
          <div style="font-size:13px;color:#8b7968;margin-bottom:8px;letter-spacing:0.05em;">Collections</div>
          <div style="font-size:42px;color:#2c1810;font-weight:300;line-height:1;">{stats.get("unique_museums",0)}</div>
        </div>
        <div>
          <div style="font-size:13px;color:#8b7968;margin-bottom:8px;letter-spacing:0.05em;">Date Range</div>
          <div style="font-size:24px;color:#2c1810;font-weight:300;margin-top:12px;line-height:1;">{stats.get("year_min")}–{stats.get("year_max")}</div>
          <div style="font-size:12px;color:#8b7968;margin-top:4px;">median: {stats.get("year_median")}</div>
        </div>
      </div>
      
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:48px;margin-top:48px;">
        <div>
          <h3 style="font-size:11px;color:#8b7968;margin-bottom:16px;letter-spacing:0.1em;font-weight:400;text-transform:uppercase;">Top Artists</h3>
          <table style="margin:0 auto;border-collapse:collapse;">
            <tbody>{ta_rows or '<tr><td style="padding:12px 24px;color:#8b7968;">No data available</td><td></td></tr>'}</tbody>
          </table>
        </div>
        
        <div>
          <h3 style="font-size:11px;color:#8b7968;margin-bottom:16px;letter-spacing:0.1em;font-weight:400;text-transform:uppercase;">Principal Collections</h3>
          <table style="margin:0 auto;border-collapse:collapse;">
            <tbody>{tm_rows or '<tr><td style="padding:12px 24px;color:#8b7968;">No data available</td><td></td></tr>'}</tbody>
          </table>
        </div>
        
        <div>
          <h3 style="font-size:11px;color:#8b7968;margin-bottom:16px;letter-spacing:0.1em;font-weight:400;text-transform:uppercase;">By Century</h3>
          <table style="margin:0 auto;border-collapse:collapse;">
            <tbody>{tc_rows or '<tr><td style="padding:12px 24px;color:#8b7968;">No data available</td><td></td></tr>'}</tbody>
          </table>
        </div>
      </div>
      
      <div style="border-bottom:1px solid #e5dfd6;margin-top:64px;"></div>
    </section>
    """


def render_gallery_html(rows: List[Dict[str, Any]], gender: str, stats: Dict[str, Any]) -> str:
    """Generate HTML gallery with academic aesthetic"""
    cards = []
    
    # Determine the correct MAX_PER_ARTIST value based on gender
    max_per_artist_display = MAX_PER_ARTIST_FEMALE if gender == "female" else MAX_PER_ARTIST_MALE
    
    if gender == "female":
        badge_text = "FEMALE ARTIST"
        badge_color = "#8b2c5e"
    else:
        badge_text = "MALE ARTIST"
        badge_color = "#1e4788"
    
    for r in rows:
        img = r.get("image_local_path", "")
        badge = f'<span style="display:inline-block;padding:3px 10px;border:1px solid {badge_color};font-size:10px;letter-spacing:0.08em;background:transparent;color:{badge_color};font-weight:400;text-transform:uppercase;font-family:\'Courier New\',monospace;">{badge_text}</span>'
        
        # TASL line
        tasl = f'Title: {html.escape(r["label"])} · Artist: {html.escape(r.get("artist_name", "Unknown"))} · Source: Wikimedia Commons · License: {html.escape(r.get("commons_license") or "—")}'
        
        year_display = r.get("inception") or "undated"
        
        cards.append(f"""
        <article style="background:#fdfcfb;border:1px solid #e5dfd6;padding:0;overflow:hidden;transition:all 0.3s ease;">
            <div style="position:relative;height:340px;background:#1a1614;display:flex;align-items:center;justify-content:center;">
                <img src="{html.escape(img)}" alt="{html.escape(r['label'])}"
                     style="max-width:100%;max-height:100%;object-fit:contain;">
            </div>
            <div style="padding:24px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;">
                    <h3 style="font-size:16px;color:#2c1810;font-weight:400;margin:0;line-height:1.3;flex:1;margin-right:12px;">{html.escape(r["label"])}</h3>
                    <div style="flex-shrink:0;">{badge}</div>
                </div>
                <div style="font-size:13px;color:#6b5d54;line-height:1.8;">
                    <div style="margin-bottom:8px;">
                        <span style="color:#8b7968;">Artist:</span> {html.escape(r.get('artist_name', 'Unknown'))}
                    </div>
                    <div style="margin-bottom:8px;">
                        <span style="color:#8b7968;">Date:</span> {year_display}
                        <span style="margin:0 8px;color:#d4c5b9;">·</span>
                        <span style="color:#8b7968;">QID:</span> {html.escape(r['qid'])}
                    </div>
                    <div style="margin-bottom:12px;">
                        <span style="color:#8b7968;">Collection:</span> {html.escape(r.get('collection') or r.get('location') or "—")}
                        {(" / "+html.escape(r.get('inventory') or "")) if r.get('inventory') else ""}
                    </div>
                    <div style="border-top:1px solid #e5dfd6;padding-top:12px;margin-top:12px;">
                        <div style="font-size:11px;">
                            <span style="color:#8b7968;">Genre:</span> {html.escape(r.get('genre') or "—")}
                            <span style="margin:0 8px;color:#d4c5b9;">·</span>
                            <span style="color:#8b7968;">Material:</span> {html.escape(r.get('material') or "—")}
                            <br>
                            {f'<span style="color:#8b7968;">Dimensions:</span> {r.get("height_cm")}×{r.get("width_cm")} cm<br>' if r.get('height_cm') else ""}
                            <a href="{html.escape(r.get('commons_license_url') or '#')}" style="color:#7c2d12;text-decoration:none;border-bottom:1px solid #d4c5b9;">{html.escape(r.get('commons_license') or 'License')}</a>
                        </div>
                    </div>
                </div>
            </div>
        </article>
        """)
    
    stats_block = render_stats_block(f"{gender.title()} Painters", stats)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Paintings by {gender.title()} Artists · Comparative Dataset</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;1,400&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            background: #f7f5f2; 
            color: #2c1810; 
            font-family: 'Crimson Text', Georgia, serif; 
            line-height: 1.6;
        }}
        h1, h2, h3 {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
        article:hover {{ 
            box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
            transform: translateY(-2px);
        }}
    </style>
</head>
<body>
    <header style="background:#fdfcfb;border-bottom:1px solid #e5dfd6;padding:48px 24px;">
        <div style="max-width:1400px;margin:0 auto;">
            <div style="text-align:center;">
                <div style="font-size:11px;letter-spacing:0.3em;color:#8b7968;margin-bottom:16px;text-transform:uppercase;">Comparative Art Historical Dataset</div>
                <h1 style="font-size:32px;color:#2c1810;font-weight:300;margin-bottom:8px;letter-spacing:-0.02em;">Paintings by {gender.title()} Artists</h1>
                <div style="font-size:13px;color:#8b7968;margin-top:16px;">Compiled {html.escape(now_iso())} · Full Commons Attribution</div>
            </div>
        </div>
    </header>
    
    <main style="max-width:1400px;margin:0 auto;padding:24px;">
        {stats_block}
        <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:24px;margin-top:48px;">
            {''.join(cards)}
        </div>
    </main>
    
    <footer style="background:#fdfcfb;border-top:1px solid #e5dfd6;margin-top:96px;padding:48px 24px;">
        <div style="max-width:1200px;margin:0 auto;text-align:center;">
            <div style="font-size:12px;color:#8b7968;line-height:1.8;">
                <div style="margin-bottom:16px;">
                    <strong>Data Source:</strong> Wikidata · <strong>Images:</strong> Wikimedia Commons · <strong>License Compliance:</strong> Full TASL Attribution
                </div>
                <div style="font-size:11px;color:#a09388;">
                    GenderPaintingsDB/3.0 · Maximum {max_per_artist_display} works per artist for collection diversity
                </div>
            </div>
        </div>
    </footer>
</body>
</html>"""
    
    return html_content


def render_overview_html(female_rows: List[Dict[str, Any]], male_rows: List[Dict[str, Any]]) -> str:
    """Generate overview HTML with academic styling"""
    all_rows = female_rows + male_rows
    all_stats = compute_stats(all_rows)
    female_stats = compute_stats(female_rows)
    male_stats = compute_stats(male_rows)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Paintings by Gender · Comparative Analysis</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;1,400&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            background: #f7f5f2; 
            color: #2c1810; 
            font-family: 'Crimson Text', Georgia, serif; 
            line-height: 1.6;
        }}
        h1, h2, h3 {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
    </style>
</head>
<body>
    <header style="background:#fdfcfb;border-bottom:1px solid #e5dfd6;padding:64px 24px;">
        <div style="max-width:1200px;margin:0 auto;text-align:center;">
            <div style="font-size:11px;letter-spacing:0.3em;color:#8b7968;margin-bottom:16px;text-transform:uppercase;">Digital Art Historical Database</div>
            <h1 style="font-size:40px;color:#2c1810;font-weight:300;margin-bottom:16px;letter-spacing:-0.02em;">Paintings by Gender</h1>
            <div style="font-size:18px;color:#6b5d54;font-weight:400;">Comparative Dataset Overview</div>
            <div style="font-size:13px;color:#8b7968;margin-top:24px;">Dataset compiled {html.escape(now_iso())}</div>
        </div>
    </header>
    
    <main style="max-width:1200px;margin:0 auto;padding:48px 24px;">
        {render_stats_block("Complete Corpus", all_stats)}
        {render_stats_block("Female Artists", female_stats)}
        {render_stats_block("Male Artists", male_stats)}
        
        <section style="margin:64px auto;padding:32px;background:#fdfcfb;border:1px solid #e5dfd6;max-width:900px;">
            <h2 style="font-size:14px;letter-spacing:0.15em;color:#7c2d12;font-weight:400;margin-bottom:24px;text-transform:uppercase;font-family:'Courier New',monospace;text-align:center;">Comparative Analysis</h2>
            
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:48px;margin-top:32px;">
                <div style="text-align:center;">
                    <h3 style="font-size:16px;color:#8b2c5e;margin-bottom:20px;font-weight:400;">Female Artists</h3>
                    <div style="margin-bottom:12px;">
                        <div style="font-size:36px;color:#2c1810;font-weight:300;line-height:1;">{len(female_rows)}</div>
                        <div style="font-size:12px;color:#8b7968;text-transform:uppercase;letter-spacing:0.05em;">Total Paintings</div>
                    </div>
                    <div style="font-size:13px;color:#6b5d54;line-height:1.8;margin-top:16px;">
                        <div><span style="color:#8b7968;">Artists:</span> {female_stats.get('unique_artists', 0)}</div>
                        <div><span style="color:#8b7968;">Date Range:</span> {female_stats.get('year_min', 'N/A')}–{female_stats.get('year_max', 'N/A')}</div>
                        <div><span style="color:#8b7968;">Median Year:</span> {female_stats.get('year_median', 'N/A')}</div>
                        <div><span style="color:#8b7968;">Licensed:</span> {sum(1 for r in female_rows if r.get('commons_license'))}</div>
                    </div>
                </div>
                
                <div style="text-align:center;">
                    <h3 style="font-size:16px;color:#1e4788;margin-bottom:20px;font-weight:400;">Male Artists</h3>
                    <div style="margin-bottom:12px;">
                        <div style="font-size:36px;color:#2c1810;font-weight:300;line-height:1;">{len(male_rows)}</div>
                        <div style="font-size:12px;color:#8b7968;text-transform:uppercase;letter-spacing:0.05em;">Total Paintings</div>
                    </div>
                    <div style="font-size:13px;color:#6b5d54;line-height:1.8;margin-top:16px;">
                        <div><span style="color:#8b7968;">Artists:</span> {male_stats.get('unique_artists', 0)}</div>
                        <div><span style="color:#8b7968;">Date Range:</span> {male_stats.get('year_min', 'N/A')}–{male_stats.get('year_max', 'N/A')}</div>
                        <div><span style="color:#8b7968;">Median Year:</span> {male_stats.get('year_median', 'N/A')}</div>
                        <div><span style="color:#8b7968;">Licensed:</span> {sum(1 for r in male_rows if r.get('commons_license'))}</div>
                    </div>
                </div>
            </div>
        </section>
        
        <section style="margin-top:64px;padding:32px;background:#fdfcfb;border:1px solid #e5dfd6;">
            <h2 style="font-size:14px;letter-spacing:0.15em;color:#7c2d12;font-weight:400;margin-bottom:24px;text-transform:uppercase;font-family:'Courier New',monospace;">Methodology</h2>
            <div style="font-size:15px;line-height:1.8;color:#6b5d54;">
                <p style="margin-bottom:16px;">This comparative dataset provides a balanced representation of paintings by artists of different genders, compiled from Wikidata with full attribution metadata from Wikimedia Commons.</p>
                
                <h3 style="font-size:13px;color:#8b7968;margin:24px 0 12px;letter-spacing:0.05em;text-transform:uppercase;">Collection Parameters</h3>
                <ul style="list-style:none;padding-left:0;">
                    <li style="margin-bottom:8px;padding-left:20px;position:relative;">
                        <span style="position:absolute;left:0;">·</span>
                        Data Source: Wikidata structured data for paintings with gender-identified artists
                    </li>
                    <li style="margin-bottom:8px;padding-left:20px;position:relative;">
                        <span style="position:absolute;left:0;">·</span>
                        Target Size: {TARGET_PER_GENDER} paintings per gender category
                    </li>
                    <li style="margin-bottom:8px;padding-left:20px;position:relative;">
                        <span style="position:absolute;left:0;">·</span>
                        Diversity Constraint: Maximum {MAX_PER_ARTIST_FEMALE} paintings per female artist, {MAX_PER_ARTIST_MALE} per male artist
                    </li>
                    <li style="margin-bottom:8px;padding-left:20px;position:relative;">
                        <span style="position:absolute;left:0;">·</span>
                        Attribution: Complete TASL (Title, Author, Source, License) metadata
                    </li>
                </ul>
                
                <h3 style="font-size:13px;color:#8b7968;margin:24px 0 12px;letter-spacing:0.05em;text-transform:uppercase;">File Organization</h3>
                <p style="font-size:14px;color:#6b5d54;">
                    Images are organized with systematic naming: GENDER_number_artist_title_year.extension<br>
                    This facilitates programmatic analysis while maintaining human readability.
                </p>
                
                <div style="margin-top:32px;padding-top:24px;border-top:1px solid #e5dfd6;">
                    <p style="font-size:13px;color:#8b7968;">
                        This dataset is designed for comparative art historical research and computational analysis,
                        with careful attention to licensing requirements and proper attribution.
                    </p>
                </div>
            </div>
        </section>
    </main>
    
    <footer style="background:#fdfcfb;border-top:1px solid #e5dfd6;margin-top:96px;padding:48px 24px;">
        <div style="max-width:1200px;margin:0 auto;text-align:center;">
            <div style="font-size:11px;color:#a09388;line-height:1.8;">
                Digital dataset compiled from Wikidata metadata and Wikimedia Commons resources<br>
                GenderPaintingsDB/3.0 · Research use with appropriate citation
            </div>
        </div>
    </footer>
</body>
</html>"""
# --- Main Function ---
def main():
    print("="*78)
    print("GENDER PAINTINGS DATABASE - COMPLETE ROBUST VERSION")
    print("With full Commons licensing, HTML galleries, and error handling")
    try:
        # If gender-specific caps exist, print them; else fall back
        _f = MAX_PER_ARTIST_FEMALE
        _m = MAX_PER_ARTIST_MALE
        print(f"Target: {TARGET_PER_GENDER} paintings per gender")
        print(f"Max per artist (F/M): {MAX_PER_ARTIST_FEMALE}/{MAX_PER_ARTIST_MALE} for diversity")
    except NameError:
        print(f"Target: {TARGET_PER_GENDER} paintings per gender")
        print(f"Max {MAX_PER_ARTIST} paintings per artist for diversity")
    print("="*78)
    
    # Check for existing work
    resume_mode = False
    if pathlib.Path("female_paintings_dataset.csv").exists() or pathlib.Path("male_paintings_dataset.csv").exists():
        print("\n⚠️  Existing dataset files detected!")
        response = input("Do you want to resume from existing progress? (y/n): ").lower().strip()
        resume_mode = response == 'y'
        if resume_mode:
            print("Resuming from existing progress...")
        else:
            print("Starting fresh (existing data will be overwritten)...")
    
    ensure_dir(OUT_DIR_FEMALE)
    ensure_dir(OUT_DIR_MALE)
    
    # Collect paintings with gender-specific caps
    try:
        female_cap = MAX_PER_ARTIST_FEMALE
        male_cap   = MAX_PER_ARTIST_MALE
    except NameError:
        female_cap = male_cap = MAX_PER_ARTIST
    
    female_paintings = collect_paintings_by_gender("female", TARGET_PER_GENDER, resume=resume_mode,
                                                   max_per_artist=female_cap)
    male_paintings   = collect_paintings_by_gender("male",   TARGET_PER_GENDER, resume=resume_mode,
                                                   max_per_artist=male_cap)
    
    # Balance the two sets BEFORE downloading/enrichment
    strategy = globals().get("BALANCE_STRATEGY", "cap_to_minority")
    fN, mN = len(female_paintings), len(male_paintings)
    print(f"\nInitial counts -> Female: {fN}, Male: {mN}")
    
    if strategy == "cap_to_minority":
        n = min(fN, mN)
        female_paintings = artist_round_robin_sample(female_paintings, n)
        male_paintings   = artist_round_robin_sample(male_paintings, n)
        print(f"Balanced by capping to minority -> {n} each")
    elif strategy == "pad_minority":
        target = max(fN, mN)
        if fN < target:
            female_paintings_extra = collect_paintings_by_gender("female", target, resume=True,
                                                                 max_per_artist=female_cap)
            female_paintings = artist_round_robin_sample(female_paintings_extra, target)
        if mN < target:
            male_paintings_extra = collect_paintings_by_gender("male", target, resume=True,
                                                               max_per_artist=male_cap)
            male_paintings = artist_round_robin_sample(male_paintings_extra, target)
        # If still not equal (supply shortage), cap both to min
        if len(female_paintings) != len(male_paintings):
            n = min(len(female_paintings), len(male_paintings))
            female_paintings = artist_round_robin_sample(female_paintings, n)
            male_paintings   = artist_round_robin_sample(male_paintings, n)
        print(f"Balanced by padding minority -> {len(female_paintings)} each")
    else:
        print("BALANCE_STRATEGY='none' — proceeding without balancing")
    
    # Download and enrich with resume capability
    print("\n" + "="*60)
    print("DOWNLOADING IMAGES & ENRICHING WITH COMMONS DATA")
    print("="*60)
    
    print("\nProcessing female paintings...")
    female_rows = enrich_and_download(female_paintings, OUT_DIR_FEMALE, "female", resume=resume_mode)
    
    print("\nProcessing male paintings...")
    male_rows = enrich_and_download(male_paintings, OUT_DIR_MALE, "male", resume=resume_mode)
    
    # Save structured data
    print("\nSaving CSV/JSON...")
    female_csv, female_json = write_csv_json(female_rows, "female")
    male_csv, male_json = write_csv_json(male_rows, "male")
    
    # Statistics
    female_stats = compute_stats(female_rows)
    male_stats = compute_stats(male_rows)
    
    # HTML galleries
    print("Generating HTML galleries...")
    female_html = render_gallery_html(female_rows, "female", female_stats)
    male_html   = render_gallery_html(male_rows, "male", male_stats)
    
    pathlib.Path("female_paintings_gallery.html").write_text(female_html, encoding="utf-8")
    pathlib.Path("male_paintings_gallery.html").write_text(male_html, encoding="utf-8")
    
    # Overview page
    overview_html = render_overview_html(female_rows, male_rows)
    pathlib.Path("paintings_by_gender_overview.html").write_text(overview_html, encoding="utf-8")
    
    # Provenance file
    prov = {
        "built_utc": now_iso(),
        "tool": "GenderPaintingsDB/3.1",
        "description": "Balanced paintings database by artist gender with full Commons licensing",
        "resume_mode": resume_mode,
        "parameters": {
            "target_per_gender": TARGET_PER_GENDER,
            "max_per_artist_female": female_cap,
            "max_per_artist_male": male_cap,
            "balance_strategy": strategy,
            "female_artists_attempted": len(get_female_artists_simple()),
            "male_artists_attempted": len(get_male_artists_simple())
        },
        "results": {
            "female": {
                "total_paintings": len(female_rows),
                "unique_artists": female_stats['unique_artists'],
                "unique_museums": female_stats['unique_museums'],
                "year_range": f"{female_stats.get('year_min', 'N/A')}-{female_stats.get('year_max', 'N/A')}",
                "with_commons_license": sum(1 for r in female_rows if r.get('commons_license'))
            },
            "male": {
                "total_paintings": len(male_rows),
                "unique_artists": male_stats['unique_artists'],
                "unique_museums": male_stats['unique_museums'],
                "year_range": f"{male_stats.get('year_min', 'N/A')}-{male_stats.get('year_max', 'N/A')}",
                "with_commons_license": sum(1 for r in male_rows if r.get('commons_license'))
            }
        },
        "notes": "Gender-balanced collection with per-artist caps and pre-download balancing."
    }
    pathlib.Path("gender_paintings_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    
    print("\n" + "="*78)
    print("COMPLETED WITH FULL FEATURES!")
    print(f"Female: {len(female_rows)} paintings from {female_stats['unique_artists']} artists")
    print(f"  With Commons license: {sum(1 for r in female_rows if r.get('commons_license'))}")
    print(f"Male: {len(male_rows)} paintings from {male_stats['unique_artists']} artists")
    print(f"  With Commons license: {sum(1 for r in male_rows if r.get('commons_license'))}")
    print(f"\nOutputs:")
    print(f"  CSV files: {female_csv}, {male_csv}")
    print(f"  JSON files: {female_json}, {male_json}")
    print(f"  HTML galleries: female_paintings_gallery.html, male_paintings_gallery.html")
    print(f"  Overview: paintings_by_gender_overview.html")
    print(f"  Provenance: gender_paintings_provenance.json")
    print(f"  Image folders: {OUT_DIR_FEMALE}/, {OUT_DIR_MALE}/")
    if resume_mode:
        print("\n✓ Run completed in RESUME mode - existing work was preserved")
    print("="*78)

if __name__ == "__main__":
    main()