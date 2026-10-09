# Marktwertupdate und Prognosekontrolle

Der tägliche Lauf verwendet pro Spieler die neueste vorhandene Marktwertbeobachtung als Prognosebasis. Eine feste Uhrzeit (bisher 22:15) entscheidet nicht mehr darüber, ob die Daten bereits aktualisiert sind.

`reports/latest.json` und `reports/history.json` enthalten:

- `market_update`: `baseline` beim ersten Lauf; `confirmed`, wenn mindestens die Hälfte der mit dem vorherigen Lauf vergleichbaren Spieler einen neueren Marktwertdatumspunkt hat; sonst `not_confirmed`. Unveränderte Werte mit neuem Datum zählen mit. Einzelne Spielerzugänge und Korrekturen am gleichen Datum bestätigen kein Update. Der Zeitpunkt beschreibt die Beobachtung im Lauf, nicht die genaue Ausführungszeit bei Kickbase.
- `live_forecast_evaluation`: Trefferquote der Richtung, MAE und RMSE der tatsächlich gespeicherten Prognosen gegen die beobachtete Veränderung am nächsten Kalendertag. Diese Kennzahlen sind getrennt von den Trainings-/Testdatenkennzahlen unter `model`.

`reports/forecast_audit.json` speichert die erste Prognose pro Spieler und Basisdatum unveränderlich. Wiederholungsläufe erzeugen keine zusätzlichen Stichproben. Eine fehlende Beobachtung am Folgetag bleibt offen; ein späterer Mehrtageswert wird nicht als Tagesergebnis gewertet. Eine korrigierte oder nicht mehr verfügbare Basis entwertet die offene Stichprobe. Prognosen, deren Ergebnis bereits in den eingelesenen Daten vorhanden ist, werden nicht nachträglich aufgenommen.

Die Kohorten werden bei der Prognose festgelegt und können sich überschneiden:

| Kohorte | Auswahl | Kennzahlen |
| --- | --- | --- |
| `traders` | Prognostizierte Veränderung positiv | Richtungsquote, MAE, RMSE |
| `flattening` | Letzte Tagesveränderung positiv, Prognose darunter | Richtungsquote, MAE, RMSE; zusätzlich Quote, ob die tatsächliche Veränderung unter der vorherigen Tagesveränderung lag |
| `losses` | Prognostizierte Veränderung negativ | Richtungsquote, MAE, RMSE |

Diese Kohorten bezeichnen Marktwerttrends, nicht die Rolle eines Spielers in der Startelf. Die Kennzahlen gelten für die gespeicherten Modellprognosen; Gebotsaufschläge, Kaufpreise und realisierte Transfergewinne sind nicht enthalten. Unbewertete Kohorten liefern `null` statt einer erfundenen Trefferquote. Es werden die letzten 90 Basisdaten mit allen zugehörigen Spielern aufbewahrt. Die Stichprobenanzahl ist immer angegeben; kleine Stichproben sind noch keine belastbare Aussage.

Die Actions-Veröffentlichung speichert die Auditdatei zusammen mit den übrigen Berichten. Der erste Lauf baut eine Basis auf; erste Trefferquoten entstehen frühestens nach der nächsten beobachteten Tagesaktualisierung. Ein Push auf `main` startet wie bisher den täglichen Workflow.

Prüfung: `python -m unittest discover -s tests -v`.
