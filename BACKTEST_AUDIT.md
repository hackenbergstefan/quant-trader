# Backtest-Kennzahlen-Audit

## 1. Umfang und untersuchte Architektur

Der Vergleichslauf verwendet die bestehende VectorBT-Engine; es wurde keine
Backtesting-Engine hinzugefügt und kein Signal, Ranking, Momentum-Gewicht,
Rebalancing oder Orderablauf geändert.

| Datei | Relevante Funktionen / Rolle |
| --- | --- |
| [momentum.py](./src/pytrading/strategies/momentum.py) | `_build_vectorbt_portfolio`, `run_variant_backtests` (Netto/Brutto mit denselben vorbereiteten Zielgewichten), `_variant_metrics`, `_holding_period_metrics`, `_position_weight_metrics`, `_covariance_based_portfolio_volatility` |
| [stats.py](./src/pytrading/stats.py) | `stats`, `calendar_cagr`; Kennzahlenpfad für `parameter_sweep` |
| [experiment_runner.py](./src/pytrading/experiment_runner.py) | Variant-Matrix, `run_comparison`, CSV/JSON/HTML-Berichte und Plots |
| [data_helper.py](./src/pytrading/data_helper.py) | Universe-Daten, EUR-Umrechnung, Kurs- und Metadaten |
| [test_momentum.py](./tests/test_momentum.py) | VectorBT-Ausführung, Volatilität, FIFO, Kosten, Exposure und Positionsgewichte |
| [test_stats.py](./tests/test_stats.py) | Übereinstimmung der Parameter-Sweep-Kennzahlen |
| [test_experiment_runner.py](./tests/test_experiment_runner.py) | Exportformat, JSON/CSV-Feldgleichheit und Report |
| [METRICS.md](./METRICS.md) | Zentrale Definitionen mit Formel, Datenquelle, Aggregation, Annualisierung und Einschränkungen |

Der neue Vergleichsbericht liegt unter
[compare-20261113T175809Z-42e5f9c425](./results/experiments/compare-20261113T175809Z-42e5f9c425/).
Der verglichene archivierte Lauf ist
[compare-20261113T171509Z-42e5f9c425](./results/experiments/compare-20261113T171509Z-42e5f9c425/).

## 2. Feststellungen

| Einstufung | Feststellung | Konsequenz / Korrektur |
| --- | --- | --- |
| **Nachgewiesener Berechnungsfehler** | `Weighted Average Portfolio Volatility` war eine gewichtete Mittelung der Einzelvolatilitäten und ignorierte Kovarianzen. | Die bestehende Kennzahl heißt nun `Weight-Weighted Average Constituent Volatility`; zusätzlich wird `Average Covariance-Based Portfolio Volatility` als `sqrt(wᵀΣw)` berichtet. |
| **Nachgewiesener Berechnungsfehler** | Fehlende Schlusskurse an Tagen, an denen ein anderes Instrument handelte, wurden in Exposure-Auswertungen effektiv als Marktwert null behandelt. Das überschätzte Cash und unterschätzte regionale Exposure. | Exposure und regionale Exposure verwenden jetzt den letzten verfügbaren Schlusskurs je Instrument. |
| **Nachgewiesener Berechnungsfehler** | FIFO-Verkäufe zählten jede Zuordnung zu einem Kauf-Lot als „Completed Trade Lot“, auch wenn vom Kauf-Lot noch Stücke offen waren. | Ein Lot zählt erst nach Verkauf seiner gesamten Kaufmenge; Teilverkäufe werden weiter FIFO zugeordnet, aber nicht als zusätzliche Lots gezählt. |
| **Nachgewiesener Definitionsfehler** | Holding- und Positionsdauer verwendeten die beobachteten Zeilen des globalen Kursindex statt der Handelssitzungen des jeweiligen Instruments. | Für bekannte Börsen werden Sitzungen aus dem zugeordneten Börsenkalender gezählt; sonst beobachtete Kurssitzungen. |
| **Nachgewiesene Reporting-Inkonsistenz** | `stats.py` verwendete VectorBTs frequenzbasierte Annualisierung, während der Vergleichsbericht CAGR nach Kalenderzeit, Sharpe mit √252 und Calmar aus derselben CAGR berechnete. | Sweep und Vergleich teilen jetzt CAGR-/Sharpe-/Calmar-Definitionen. `Max Drawdown` war bereits konsistent. |
| **Reporting-Erweiterung** | Es gab keinen JSON-Export der Ergebniszeilen. Zudem waren Definitionen nicht je Lauf beigefügt. | `results.json` exportiert dieselben Ergebnisfelder wie CSV/HTML; jeder Lauf enthält `metric_definitions.md`. Nicht-endliche JSON-Werte werden als `"Infinity"`/`"-Infinity"` kodiert. |
| **Kein Fehler festgestellt** | Gebühren und Slippage werden im Netto-VectorBT-Lauf verrechnet und im Bericht aus den tatsächlichen Orderrecords ausgewiesen. | Es erfolgt keine zweite Kostenbelastung; Gebühren- und Slippage-Berechnung wurden mit Kauf und Verkauf getestet. |
| **Kein Fehler festgestellt** | Zielgewichte werden beim regulären Rebalancing auf die konfigurierte Positionsgrenze begrenzt; tatsächliche Gewichte werden separat aus Marktwert/NAV gemessen. | Die Begrenzungslogik blieb unverändert. Die Kennzahl `Time at or Above Position Cap` bezieht sich auf tatsächliche Gewichte und erzwingt keine untermonatlichen Verkäufe. |

## 3. Korrigierte Berechnungen

- **Kovarianzbasierte Volatilität:** Für jedes Datum wird die aktuelle tatsächliche
  Gewichtung der gehaltenen Positionen verwendet. Die Kovarianz wird aus den
  letzten höchstens 60 gemeinsam beobachteten Tagesrenditen der an diesem Datum
  gehaltenen Instrumente berechnet; mindestens 20 vollständige Beobachtungen
  sind erforderlich. Die Schätzung endet am jeweiligen Datum, wird mit √252
  annualisiert und über definierte Tageswerte gemittelt. Cash trägt Rendite
  null bei. Fehlende Beobachtungen werden nicht vorwärtsgefüllt; bei
  unterschiedlichen Börsenkalendern gehen nur gemeinsame Renditebeobachtungen
  in die Kovarianz ein.
- **FIFO-Lots:** Für Käufe von 10 und 5 Aktien mit Verkäufen von 8 und 7 Aktien
  werden genau zwei Lots abgeschlossen, nicht drei Verkaufsteilzuordnungen.
  Die Dauer eines Lots endet mit dem Verkauf seiner letzten verbleibenden
  Stücke; die gewichtete Dauer verwendet die ursprüngliche Kaufmenge.
- **Exposure:** Tagesmarktwerte verwenden pro Instrument den letzten
  verfügbaren Schlusskurs. Durchschnittliches Exposure, Cash Allocation und
  regionale Exposures werden aus derselben Marktwertbasis abgeleitet.
- **CAGR / Sharpe / Calmar:** CAGR basiert auf verstrichenen Kalendertagen
  (`365.25` Tage pro Jahr, Mindestdauer `1/252` Jahr), Sharpe auf Stichproben-
  Standardabweichung täglicher Renditen und √252, Calmar auf CAGR geteilt durch
  den absoluten maximalen Drawdown.
- **Orderkosten und Endbewertung:** Proportionale Gebühren beruhen auf
  ausgeführter Menge und Fill-Preis; Fixgebühren werden aus den aufgezeichneten
  Ordergebühren abgeleitet; Slippage verwendet die Abweichung vom
  Ausführungskurs. Offene Positionen werden am letzten Bewertungsdatum im
  VectorBT-NAV bewertet und nicht als abgeschlossene Lots/Positionen gezählt.

## 4. Kennzahlendefinitionen und Änderungen

Die vollständigen Definitionen stehen in [METRICS.md](./METRICS.md). Die
Ergebnisfelder in CSV, JSON und HTML stammen aus denselben abgeflachten
Ergebniszeilen. `config.json` bleibt ein Konfigurationsarchiv, kein zweiter
Kennzahlenexport.

Die alte Kennzahl `Weighted Average Portfolio Volatility` wurde nicht
numerisch neu definiert: ihre Zahlen erscheinen unverändert unter
`Weight-Weighted Average Constituent Volatility`. Die neue
`Average Covariance-Based Portfolio Volatility` ergänzt sie und berücksichtigt
Korrelationen.

## 5. Tests und Regression

Der vollständige Lauf `uv run pytest -q` ergab **60 passed**. Die 57 Warnungen
stammen aus bereits verwendeten VectorBT-/Pandas-Deprecations; es gab keine
Testfehler. Ruff-Prüfung und Format-Check waren erfolgreich.

Abgedeckt sind insbesondere:

- einzelne Position, perfekt positive Korrelation, Unkorreliertheit,
  perfekte negative Korrelation, Cash-Anteil, wechselnde Holdings,
  fehlende Renditen, Börsenfeiertage und ein Look-ahead-Regressionstest;
- vollständige und partielle FIFO-Verkäufe, mehrere Käufe, offene Lots,
  mehrere Instrumente und börsenkalenderbasierte Haltedauer;
- tatsächliche Gewichte oberhalb des Zielgewichtslimits, HHI,
  Exposure/Cash-/Regionalsummen;
- nachvollziehbare Kauf-/Verkaufskosten einschließlich proportionaler und
  fixer Gebühren sowie Slippage;
- identische Ergebnisfeldnamen in CSV/JSON und der HTML-Ergebnistabelle;
- Kalender-CAGR sowie die einheitlichen Sweep-Kennzahlen.

## 6. Vergleichslauf und Abweichungen

Der neue Lauf wurde nach Commit `a1dd3f146dae2b4b7af47cd59ca1ca2a936eae68`
ausgeführt. Parameter: 2010-01-01 bis 2017-01-01, EUR 3.000 Startkapital,
0,10 % proportional, EUR 1 Fixgebühr, 0,05 % Slippage, 1.101 Instrumente.
Der Universe-Fingerprint stimmt mit dem archivierten Vergleich überein; alle
vier Varianten waren erfolgreich. CSV, JSON und HTML enthalten dieselben 111
Felder und vier Varianten.

### Unveränderte Strategie- und Kostenresultate

| Variante | Net Total Return | Gross Total Return | Transaction Costs |
| --- | ---: | ---: | ---: |
| AB-M5 | 247,5825 % | 377,0147 % | EUR 1.474,48 |
| ABC-M5 | 268,8204 % | 403,4725 % | EUR 1.509,27 |
| AB-M10 | 202,9899 % | 359,8688 % | EUR 1.908,06 |
| ABC-M10 | 120,9438 % | 275,5995 % | EUR 2.004,38 |

Netto-/Brutto-Endwerte und Renditen, CAGR, Drawdown, Sharpe, Sortino,
Profit-Factor, Orderzahlen, Turnover und sämtliche Kostenfelder sind gegenüber
dem Archivlauf unverändert. Die tatsächlichen Gewichte, HHI und
`Time at or Above Position Cap` blieben ebenfalls unverändert.

### Exposure und Cash: archiviert → korrigiert

| Variante | Average Exposure | Cash Allocation | Time with Cash |
| --- | ---: | ---: | ---: |
| AB-M5 | 81,6499 % → 84,3171 % | 18,3501 % → 15,6829 % | 28,3050 % → 24,5200 % |
| ABC-M5 | 81,7418 % → 84,4172 % | 18,2582 % → 15,5828 % | 26,2205 % → 22,1064 % |
| AB-M10 | 81,9687 % → 84,6345 % | 18,0313 % → 15,3655 % | 30,0055 % → 24,9040 % |
| ABC-M10 | 81,8307 % → 84,4464 % | 18,1693 % → 15,5536 % | 22,6550 % → 16,9501 % |

Die durchschnittlichen regionalen Exposures änderten sich wie folgt:

| Variante | EUROPE | USA | OTHER |
| --- | ---: | ---: | ---: |
| AB-M5 | 31,8742 % → 32,8090 % | 45,8310 % → 47,4511 % | 3,9446 % → 4,0569 % |
| ABC-M5 | 31,9842 % → 32,9269 % | 45,7758 % → 47,3943 % | 3,9818 % → 4,0960 % |
| AB-M10 | 32,2626 % → 33,2216 % | 46,6694 % → 48,2900 % | 3,0367 % → 3,1229 % |
| ABC-M10 | 34,9319 % → 35,9520 % | 44,1076 % → 45,6281 % | 2,7911 % → 2,8663 % |

### FIFO- und Haltedauermetriken: archiviert → korrigiert

| Variante | Completed Trade Lots | Avg Holding Period | Median Holding Period | Weighted Avg Holding Period | Avg Position Duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| AB-M5 | 300 → 300 | 24,5433 → 24,5167 | 20 → 20 | 27,6779 → 27,6360 | 24,5433 → 24,5167 |
| ABC-M5 | 300 → 300 | 24,5433 → 24,5167 | 20 → 20 | 27,7406 → 27,7017 | 24,5433 → 24,5167 |
| AB-M10 | 555 → 555 | 26,1820 → 26,1351 | 20 → 20 | 30,2079 → 30,1509 | 26,1820 → 26,1351 |
| ABC-M10 | 729 → 626 | 26,4856 → 26,3403 | 20 → 20 | 26,8643 → 28,6500 | 26,2282 → 26,1836 |

Alle Werte sind Sitzungen. Bei ABC-M10 wurden 103 zuvor als „abgeschlossen“
gezählte Teilzuordnungen nicht als abgeschlossene Kauf-Lots gezählt; die 561
abgeschlossenen Positionsepisoden blieben unverändert. Median-Haltedauer,
offene Lots/Positionen und deren Durchschnittsdauer blieben unverändert.

### Volatilität: neue Definitionen

| Variante | Average Realized Volatility | Weight-Weighted Average Constituent Volatility | Average Covariance-Based Portfolio Volatility |
| --- | ---: | ---: | ---: |
| AB-M5 | 25,4127 % | 47,7060 % | 24,7400 % |
| ABC-M5 | 25,4752 % | 47,7756 % | 24,7877 % |
| AB-M10 | 20,5937 % | 42,7161 % | 19,6529 % |
| ABC-M10 | 19,5044 % | 38,6968 % | 17,5399 % |

Die beiden ersten Spalten behalten ihre bisherigen Zahlen und Bedeutungen
(realisiertes Portfolio-Risiko bzw. Einzelvolatilitäten ohne Kovarianz). Die
neue dritte Spalte ist die tatsächliche, kovarianzbasierte Schätzung und daher
nicht mit der gewichteten Einzelvolatilität gleichzusetzen.

## 7. Nicht behobene Einschränkungen / offene Punkte

- Die Instrumentendateien enthalten keine historischen Indexmitgliedschaften;
  Survivorship Bias bleibt möglich.
- Kovarianzschätzungen benötigen mindestens 20 gemeinsame Beobachtungen im
  60-Beobachtungsfenster. Bei dünnen Daten oder wechselnden Kalendern können
  aktive Tage daher ohne Schätzung bleiben. Börsen ohne erkannte Kalender-ID
  verwenden beobachtete Kursdaten; fehlende Werte und Börsenfeiertage lassen
  sich dort nicht immer unterscheiden.
- Die Kovarianzkennzahl ist eine rollierende tägliche Schätzung auf
  Schlusskursbasis, kein intraday Risikomaß. Renditen verschiedener Börsen
  werden nur an gemeinsamen Beobachtungstagen kovariiert.
- Die Resultate wurden deterministisch mit demselben lokalen Daten-Fingerprint,
  aber nicht gegen eine externe unabhängige Kursquelle validiert.

Es wurden keine verbleibenden Fehler in Kostenverbuchung, Brutto-/Netto-
Auswertung, Positionsgewichtslimit oder Orderausführung nachgewiesen.

## 8. Auswirkungen auf frühere Backtests

Historische Ergebnisordner wurden nicht überschrieben. Bei identischen
Strategieparametern bleiben Portfolio-Performance, Ausführung und Kosten
unverändert; korrigiert bzw. ergänzt wurden Exposure-Auswertungen, FIFO-
Definitionen, Haltedauern und Volatilitätsdarstellung. Alte Berichte führen
weiterhin den irreführenden Feldnamen `Weighted Average Portfolio Volatility`;
für neue Läufe gilt stattdessen die in [METRICS.md](./METRICS.md) dokumentierte
Bezeichnung und zusätzliche Kovarianzkennzahl. Der Parameter-Sweep verwendet
nun dieselben CAGR-/Sharpe-/Calmar-Definitionen wie der Vergleichsbericht.
