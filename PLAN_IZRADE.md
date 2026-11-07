# Plan izrade demoa: Data Refinery

**Rok:** pitch nedelja 8. 11. u **18:00**. **Code freeze:** 13:00.
**Mašina:** MacBook Air M2, 8 GB RAM, PyCharm + Claude Code. **Projekat:** `~/Desktop/Demo/refinery`.
**Tim:** Marko (labele, izvoz, integracija), Zoran (QC kapije), Dušan (prijem, privatnost, dashboard).

Kostur koda postoji i radi od početka do kraja na sintetičkom snimku. Ovaj plan ga vodi do demoa na **pravim** snimcima kroz 8 checkpointa (CP). Svaki CP ima:
- **Cilj**: šta posle njega postoji.
- **Zadaci**: šta tačno treba uraditi.
- **Gotovo kad**: proverljiv kriterijum (komanda + očekivani izlaz).
- **Prompt za Claude Code**: kako da se zada, ako radiš preko njega.

Pravilo: **CP se ne zatvara dok kriterijum "Gotovo kad" ne prođe.** Posle svakog CP-a: `git commit -m "CPx: ..."`.

---

## Vremenski raspored

| Vreme | CP | Ko |
|---|---|---|
| sub 22:30–23:15 | CP0 Okruženje + Claude Code u PyCharmu | Marko |
| sub 23:15–00:30 | CP1 Prvi pravi snimak (MediaPipe ruke + lica) | Marko + Dušan |
| sub 23:15–01:00 | CP2 IndustReal snimak + anotacije ishoda | Zoran |
| ned 08:30–10:00 | CP3 Kalibracija QC kapija | Zoran |
| ned 08:30–10:30 | CP4 Jezičke labele (VLM 3B) | Marko |
| ned 10:00–11:30 | CP5 LeRobot izvoz i provera | Marko |
| ned 08:30–12:00 | CP6 Dashboard za priču pitcha | Dušan |
| ned 12:00–13:00 | CP7 Freeze: keš, rezervni video, generalna proba | svi |
| ned 13:00–18:00 | prezentacija + probe (bez novih funkcija!) | svi |

Ako nešto kasni: seci po prioritetu iz odeljka "Šta se seče" na kraju.

---

## CP0: Okruženje (Marko, ~45 min)

**Cilj:** projekat radi u PyCharmu, Claude Code je povezan sa IDE-om, sintetički test prolazi.

**Zadaci**
1. PyCharm → *Settings → Project → Python Interpreter* → izaberi `refinery/.venv/bin/python` (Python 3.12).
2. U PyCharm terminalu (sa aktivnim `.venv`) proveri OpenCV. Instalirani su i `opencv-python` i `opencv-contrib-python`, a oba pišu u isti `cv2` folder:
   ```bash
   python -c "import cv2, mediapipe; print(cv2.__version__, mediapipe.__version__)"
   ```
   Ako baci grešku: `pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python`. MediaPipe traži contrib verziju.
3. Instaliraj Claude Code CLI i plugin (uputstvo je u odeljku "Claude Code u PyCharmu" na dnu).
4. `git init && git add . && git commit -m "CP0: kostur"`. `.gitignore` već isključuje `runs/`, modele i video.
5. Sintetički test:
   ```bash
   python run_pipeline.py --video data/synth_h264.mp4 --annotations data/synth_steps.csv --faces none --hands fake --name synth --no-lerobot
   streamlit run app.py
   ```

**Gotovo kad:** terminal ispiše `5/5 pakovanje` i `gotovo`, dashboard prikaže snimak `synth` sa **Upotrebljivo 60 %** i 2 epizode, a `claude` u PyCharm terminalu prikaže povezanost sa IDE-om.

**Prompt:** `Pročitaj CLAUDE.md i PLAN_IZRADE.md. Pokreni sintetički test iz CP0 i reci mi da li prolazi. Ništa ne menjaj.`

---

## CP1: Prvi pravi snimak (Marko + Dušan, ~75 min)

**Cilj:** pipeline radi sa **pravim** MediaPipe rukama i blurom lica na snimku koji ste sami napravili.

**Zadaci**
1. **Snimanje (Dušan):** 60 s telefonom pričvršćenim na glavu ili grudi (kaiš, kapa sa trakom). Kadar horizontalno, 1080p/30 fps. Sadržaj:
   - 0–40 s: montažni zadatak rukama (npr. šrafljenje, LEGO, sklapanje dela), ruke stalno u kadru;
   - u jednom trenutku kolega uđe u kadar (lice!) i vidi se ekran laptopa;
   - 3–5 s namerno trese kameru, 3–5 s pokrije sočivo ili ugasi svetlo, 3–5 s ruke van kadra.
   Tako svaka QC kapija ima šta da uhvati.
2. Prebaci snimak u `refinery/data/tim_01.mp4`.
3. Pokreni:
   ```bash
   python run_pipeline.py --video data/tim_01.mp4 --name tim_01 --faces mediapipe --hands mediapipe --max-seconds 60 --no-lerobot
   ```
4. **Proveri levu i desnu ruku u `overlay.mp4`:** narandžasta = leva, plava = desna (iz ugla nosioca). U `hands.py` je labela MediaPipe-a obrnuta, jer MediaPipe pretpostavlja selfie sliku. Ako su boje i dalje pogrešne, ukloni zamenu u `MediaPipeHands.detect`.
5. Proveri da je lice zamućeno u svakom frejmu gde se pojavljuje. Ako promakne, spusti `min_conf` u `MediaPipeFaces` (0.4 → 0.3) i/ili povećaj `hold_frames` u `Anonymizer`.
6. Izmeri brzinu: koliko sekundi traje obrada 60 s snimka (log ima vremena). Ako je >3 min, dodaj smanjivanje rezolucije za detekciju ruku (npr. 960 px širine).

**Gotovo kad:**
- KPI "Frejmova sa rukom" je ≥ 70% u dashboardu;
- lice nije čitljivo ni u jednom frejmu overlay-a;
- boje ruku odgovaraju strani;
- obrada 60 s traje < 3 min.

**Prompt:** `CP1: pokreni pipeline na data/tim_01.mp4 (komanda iz plana). Zatim iz runs/tim_01/frames.parquet izračunaj udeo frejmova sa rukom, broj frejmova sa licem i trajanje obrade. Predloži izmene samo ako kriterijum CP1 ne prolazi.`

---

## CP2: IndustReal snimak + anotacije ishoda (Zoran, ~90 min)

**Cilj:** industrijski snimak sa **labelama grešaka** ide kroz pipeline. Na vremenskoj osi se vide koraci, a greške su crvene.

**Zadaci**
1. Preuzmi **jedan** snimak sa 4TU: https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224 (Apache-2.0). Ne ceo skup od 6 h.
2. Pogledaj strukturu: `find <snimak> -maxdepth 2 | head -50` i prve redove fajlova sa anotacijama (PSR / greške / akcije). **Format nismo proverili**, pa je ovo ključni korak.
3. Prilagodi `scripts/industreal_to_csv.py` (`COLUMN_MAP` i čitanje fajla) da napravi:
   - `data/ir_01.mp4` iz RGB frejmova (10 fps),
   - `data/ir_01_steps.csv` sa kolonama `start_s,end_s,step,outcome,error_type`.
4. Pokreni:
   ```bash
   python run_pipeline.py --video data/ir_01.mp4 --annotations data/ir_01_steps.csv --name ir_01 --faces mediapipe --hands mediapipe --hfov 64 --max-seconds 120 --no-lerobot
   ```
5. Izaberi isečak od 60–120 s koji ima **bar jednu grešku i oporavak**. To je glavni deo demoa.

**Gotovo kad:** tab *Labele* prikazuje bar 3 koraka sa nazivima iz IndustReal-a i bar jedan crveni (`error`) segment, a `episodes.json` ima `success: false` i `first_error_s` za tu epizodu.

**Prompt:** `CP2: evo izlaza find komande i prvih redova anotacija: <nalepi>. Prilagodi scripts/industreal_to_csv.py tome i napravi data/ir_01.mp4 i data/ir_01_steps.csv.`

---

## CP3: Kalibracija QC kapija (Zoran, ~90 min)

**Cilj:** pragovi imaju smisla na pravim snimcima, a svaka kapija se bar jednom vidi kako radi.

**Zadaci**
1. U dashboardu (sidebar → *QC pragovi*) podesi pragove na `tim_01` i `ir_01`. Cilj je da namerni defekti iz CP1 budu odbijeni, a normalan rad prihvaćen.
2. Za svaku kapiju zapiši gde je okinula (segment, vreme). Ako neka nikad ne okida, napravi kratak snimak koji je izaziva.
3. Prepiši dobre vrednosti u `DEFAULT_CFG` u `refinery/qc.py`.
4. *(Opciono)* Za IndustReal dodaj kapiju "sinhronizacija": razlika timestampova RGB ↔ HL2 ruke, ako fajlovi to omogućavaju.
5. Proveri ručnu trijažu: tab *Ručna trijaža* → odobri ili odbij 2 segmenta → *Ponovo spakuj* → procenat se menja.

**Gotovo kad:** na `tim_01` odbijeni segmenti tačno odgovaraju namernim defektima (±1 segment), na `ir_01` upotrebljivo je ≥ 60%, i svaka od 6 kapija se pojavljuje bar jednom u nekom od snimaka.

**Prompt:** `CP3: učitaj runs/tim_01/frames.parquet i segments.json. Za svaku QC metriku prikaži raspodelu (min/medijana/max) i predloži pragove tako da su segmenti u ovim intervalima odbijeni: <vremena defekata>. Ne menjaj DEFAULT_CFG dok ne potvrdim.`

---

## CP4: Jezičke labele, VLM (Marko, ~2 h)

**Cilj:** podzadaci imaju smislene opise na engleskom, generisane lokalno.

**Zadaci**
1. `pip install mlx-vlm`.
2. **8 GB RAM:** u `refinery/label.py` promeni default u `MLXCaptioner` na `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`. 7B je preveliki. Dok VLM radi, zatvori Chrome i ostale teške aplikacije.
3. Izolovan test na jednom kolažu (`runs/tim_01/collages/seg_0000.jpg`): koliko sekundi traje jedan opis i da li poštuje format `<verb> <object> [to <destination>]`.
4. Ako je > 8 s po segmentu, opisuj samo svaki drugi segment ili samo segmente koji su prošli QC (već je tako). Spoji uzastopne iste opise.
5. Pokreni `--captions mlx` na `tim_01`.
6. *(Ako ostane vremena)* **tačnost labela:** na `ir_01` uporedi VLM opis sa nazivom koraka iz IndustReal-a (poklapanje glagola i objekta) i prikaži procenat u dashboardu. To je argument "kvalitet kao proizvod".

**Gotovo kad:** na `tim_01` ≥ 80% prihvaćenih segmenata ima opis iz VLM-a koji počinje glagolom sa liste `VERBS`, a obrada opisa za 60 s snimka traje < 5 min.

**Prompt:** `CP4: napravi skriptu scripts/test_caption.py koja učitava MLXCaptioner sa Qwen2.5-VL-3B-Instruct-4bit, opisuje 3 kolaža iz runs/tim_01/collages i ispisuje vreme po kolažu. Pokreni je.`

---

## CP5: LeRobot izvoz i provera (Marko, ~60 min)

**Cilj:** paket za kupca je pravi LeRobot v3 skup koji se učitava zvaničnim API-jem.

**Zadaci**
1. `pip install "lerobot[dataset]>=0.4"` (povlači torch, trajaće par minuta).
2. Pokreni pipeline bez `--no-lerobot` na `ir_01` (ili *Ponovo spakuj* u dashboardu).
3. Provera:
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset("refinery/ir_01", root="runs/ir_01/export/lerobot")
   print(ds.num_episodes, ds.num_frames, ds.fps); x = ds[0]; print({k: getattr(v, "shape", v) for k, v in x.items()})
   ```
4. Sačuvaj ovu proveru kao `scripts/verify_lerobot.py`. Na pitchu se pokreće uživo: "kupac učitava jednom linijom".

**Gotovo kad:** `verify_lerobot.py` ispisuje broj epizoda > 0, `observation.state (142,)`, `action (138,)`, `action.gripper (14,)`, `observation.images.ego (3, H, W)` i `task` sa nazivom koraka. U tabu *Isporuka* se vidi `info.json`, a zip se preuzima.

**Prompt:** `CP5: napravi scripts/verify_lerobot.py po planu, pokreni ga na runs/ir_01 i ispravi izvoz ako učitavanje pukne.`

---

## CP6: Dashboard za priču pitcha (Dušan, paralelno sa CP3–CP5)

**Cilj:** dashboard prati redosled priče, a ne redosled koda.

**Zadaci**
1. Redosled tabova po priči: **Pipeline** (sirovo → obrađeno) → **Kvalitet** (levak, razlozi) → **Labele** (koraci, greška crveno, ruke) → **Isporuka** (LeRobot, manifest, zip). Trijaža ostaje, ali kao poslednji tab.
2. Gornji KPI red je ono što žiri pamti: *Upotrebljivo %*, *Zamućenih lica*, *Epizoda za kupca*, *Segmenata sa greškom*. Dodaj ovaj poslednji iz `m.labels["segments_with_error"]`.
3. Sidebar *Prijem novog snimka* mora da radi uživo sa kratkim klipom (15–20 s), da se na pitchu pokaže prijem.
4. Jedan panel "Šta kupac dobija" (statičan tekst + `features.json`): dimenzije stanja i akcija, oba rasporeda (dexterous / gripper), poreklo podataka.
5. Jezik UI-ja: usaglasiti sa jezikom pitcha (EN ili SR) i držati se jednog.

**Gotovo kad:** neko ko ne zna projekat prođe kroz 4 taba za < 3 min i ume da kaže šta je proizvod. Prijem klipa od 20 s uživo radi za < 1 min.

**Prompt:** `CP6: u app.py preuredi tabove u redosled Pipeline, Kvalitet, Labele, Isporuka, Trijaža i dodaj KPI "Segmenata sa greškom". Prikaži mi diff pre izmene.`

---

## CP7: Freeze i rezerve (svi, 12:00–13:00)

**Zadaci**
1. `git commit -m "CP7: freeze"` + `git tag demo-v1`. Posle ovoga samo ispravke bagova.
2. **Keš:** unapred obradi `ir_01`, `tim_01` i kratak klip za uživo prijem. Na pitchu se otvaraju gotovi runovi, a uživo se pokreće samo kratki klip.
3. **Rezervni video:** QuickTime → *New Screen Recording*, snimi ceo prolaz kroz dashboard (2–3 min).
4. Generalna proba sa merenjem vremena. Isključi Wi-Fi da proveriš da sve radi offline (modeli su lokalni).
5. Napuni laptop, isključi notifikacije, `streamlit run app.py` pokreni pre izlaska na binu.

**Gotovo kad:** demo prođe 2 puta zaredom bez greške i postoji rezervni video.

---

## Šta se seče (ako kasni), redom
1. Tačnost labela (CP4.6) i kapija sinhronizacije (CP3.4)
2. VLM jezik → ostaje `template` + nazivi koraka iz IndustReal anotacija (jezik iz anotacija je već dovoljan za demo)
3. Blur teksta i ekrana (OCR): na pitchu reći "sledeći korak"
4. Uživo prijem → samo gotovi runovi + rezervni video

**Ne seče se:** blur lica, QC levak sa razlozima, labele ishoda (greška crveno), LeRobot izvoz. To su četiri stuba priče.

---

## Claude Code u PyCharmu

**1. Instalacija CLI-ja** (u macOS Terminalu):
```bash
curl -fsSL https://claude.ai/install.sh | bash
# otvori NOVI terminal:
claude --version
```
Ako javi `command not found: claude`, uradi PATH korak koji installer ispiše na kraju (*Setup notes*).

**2. Plugin:** PyCharm → *Settings → Plugins → Marketplace* → instaliraj **"Claude Code [Beta]"** (izdavač Anthropic) → restartuj PyCharm. Postoje i neoficijalni pluginovi sa sličnim imenom. Uzmi zvanični.

**3. Pokretanje:** otvori projekat `refinery`. U PyCharm terminalu (*View → Tool Windows → Terminal*), u korenu projekta:
```bash
claude
```
Prvi put traži prijavu (Claude nalog sa plaćenim planom, bez API ključa). Brzo otvaranje: **Cmd+Esc**.

**4. Šta dobijaš u PyCharmu**
- Izmene se otvaraju u PyCharm **diff prikazu**: pregledaš i prihvatiš.
- Selektovan kod ili otvoren fajl automatski ide Claudu kao kontekst.
- **Cmd+Option+K** ubacuje referencu na fajl ili linije (npr. `@refinery/qc.py#L40-80`).
- Ako ESC ne prekida Claudea: *Settings → Tools → Terminal* → isključi "Move focus to the editor with Escape".
- Ako diffovi idu u terminal umesto u IDE: u Claude Code `/config` → **Diff tool** = `auto`.

**5. Način rada po checkpointima**
- Na početku svakog CP-a: `/clear` (čist kontekst), pa prompt iz plana koji počinje sa "CPx:".
- Za veće izmene prvo **plan mode** (Shift+Tab dok ne piše *plan mode*): Claude predloži plan, ti odobriš, pa tek onda menja kod.
- Ostavi ručno odobravanje izmena. Dokumentacija upozorava da auto-prihvatanje u JetBrains-u može da menja IDE konfiguraciju bez pitanja.
- Na kraju CP-a: `Pokreni kriterijum "Gotovo kad" za CPx i napravi git commit ako prolazi.`
- `CLAUDE.md` u `refinery/` Claude čita automatski: tamo su pravila projekta i komande.

Izvori: https://code.claude.com/docs/en/jetbrains , https://code.claude.com/docs/en/quickstart
