Erläuterung der PPO- und Simulationslogs

Diese Logs liefern detaillierte Rückmeldungen darüber, wie dein Reinforcement-Learning-Agent (PPO) während des Trainings und in der finalen Simulation agiert. Im Folgenden erklären wir die einzelnen Abschnitte:


1. Evaluations-Logs

Evaluationslogs werden periodisch ausgegeben, um zu überprüfen, wie gut der Agent bereits performt. Sie enthalten:
	•	mean_ep_length
	•	Bedeutung: Durchschnittliche Länge einer Episode in Evaluationsläufen.
	•	Beispiel: Ein Wert von 2 bedeutet, dass in der Evaluationsumgebung jede Episode (d. h. ein kompletter Durchlauf) im Durchschnitt 2 Schritte benötigt.
	•	mean_reward
	•	Bedeutung: Durchschnittliche Belohnung, die der Agent pro Episode erzielt.
	•	Beispiel: Ein Wert von -552 zeigt, dass der Agent in der Evaluierung einen konstanten (wenn auch negativen) Reward erhält. Das dient als Referenz, um Verbesserungen im Training zu messen.
	•	total_timesteps
	•	Bedeutung: Die Anzahl der insgesamt verarbeiteten Simulationsschritte.
	•	Beispiel: 6000 oder 7000 zeigt, wie weit das Training bereits fortgeschritten ist.



2. Rollout- und Trainingslogs

Diese Logs geben Aufschluss über die Interaktionen des Agenten mit der Umgebung während des Sammelns von Erfahrungen (Rollouts) und der anschließenden Policy-Aktualisierung:
	•	ep_len_mean (Rollout)
	•	Bedeutung: Durchschnittliche Episodenlänge während der Rollouts.
	•	Beispiel: Werte wie 3.28, 2.88 oder 2.68 zeigen, wie viele Schritte der Agent in einer Episode benötigt – tendenziell sinkt dieser Wert, wenn der Agent schneller lernt, Aufgaben zu beenden.
	•	ep_rew_mean (Rollout)
	•	Bedeutung: Durchschnittliche Belohnung pro Episode in den Rollouts.
	•	Beispiel:
	•	Ein anfänglicher Wert von -1440 signalisiert, dass der Agent zu Beginn ineffiziente Entscheidungen trifft (z. B. hohe Produktionszeiten oder Kosten).
	•	Mit der Zeit steigen die Werte (z. B. 602 oder 932), was darauf hinweist, dass der Agent bessere (effizientere) Strategien entwickelt.
	•	fps (Frames per Second)
	•	Bedeutung: Anzahl der Simulationsschritte pro Sekunde – ein Indikator für die Rechengeschwindigkeit.
	•	Beispiel: 689 fps zeigt, dass viele Schritte in kurzer Zeit simuliert werden.
	•	Iterations, time_elapsed, total_timesteps
	•	Bedeutung:
	•	Iterations: Anzahl der abgeschlossenen Rollout-Perioden.
	•	time_elapsed: Zeit in Sekunden, die seit Trainingsbeginn vergangen ist.
	•	total_timesteps: Kumulative Simulationsschritte, die bisher gesammelt wurden.
	•	Beispiel: In einer Iteration wurden z. B. 2048 Schritte in 3 Sekunden ausgeführt.
	•	Trainingsmetriken (train/…)
	•	approx_kl:
	•	Misst, wie stark sich die Policy zwischen Updates ändert. Kleine Werte (z. B. 0.0095) bedeuten stabile Updates.
	•	clip_fraction:
	•	Der Anteil der Daten, die vom Clipping (zum Schutz vor zu großen Policy-Updates) betroffen sind. Ein Wert von 0.0875 bedeutet, dass etwa 8,75 % der Samples beschnitten wurden.
	•	entropy_loss:
	•	Ein Maß für die Förderung von Exploration; negative Werte (z. B. -4.09) helfen dabei, zu deterministische Entscheidungen zu vermeiden.
	•	explained_variance:
	•	Gibt an, wie gut die Value-Funktion die zukünftigen Belohnungen vorhersagen kann.
	•	learning_rate:
	•	Die Lernrate, hier konstant bei 0.0001.
	•	loss, n_updates, policy_gradient_loss, value_loss:
	•	Diese Werte geben den Gesamtverlust, die Anzahl der Updates sowie die spezifischen Verluste der Policy- und Value-Funktion an. Sinkende Verlustwerte deuten auf eine Verbesserung hin.


3. Simulation Step Summaries

Diese Abschnitte zeigen, wie die Produktionssimulation in einzelnen Schritten abläuft:
	•	Time:
	•	Bedeutung: Aktuelle Simulationszeit in Zeiteinheiten.
	•	Beispiel: Time: 197.00 bedeutet, 197 Zeiteinheiten sind seit Start vergangen.
	•	Action:
	•	Bedeutung: Die vom Agenten getroffene Aktion als Tupel, z. B. [2 18].
	•	Beispiel:
	•	2 könnte für den SPT-Modus (Shortest Processing Time) stehen.
	•	18 bedeutet, dass der Agent plant, (18+1 = 19) Jobs gleichzeitig zu planen.
	•	Reward:
	•	Bedeutung: Die Belohnung, die der Agent in diesem Schritt erhält.
	•	Beispiel: Reward: -193.001 zeigt, dass in diesem Schritt Kosten oder lange Produktionszeiten aufgetreten sind.
	•	Finished Jobs und Waiting Jobs:
	•	Bedeutung:
	•	Finished Jobs: Anzahl der Jobs, die in diesem Schritt abgeschlossen wurden.
	•	Waiting Jobs: Anzahl der Jobs, die nach diesem Schritt noch in der Warteschlange stehen.
	•	Beispiel:
	•	Finished Jobs: 1, Waiting Jobs: 11 zeigt, dass 1 Job abgeschlossen wurde und noch 11 Jobs warten.
	•	Total Cost:
	•	Bedeutung: Die bis zum aktuellen Zeitpunkt angefallenen Produktionskosten.
	•	Beispiel: Total Cost: 0 oder 205 – je nachdem, ob bereits Zusatzkosten (z. B. für Rüstzeiten oder Lagerkosten) entstanden sind.
	•	Reorder Events:
	•	Bedeutung: Zeigt, ob und wann Produktionshilfsmittel (z. B. “Schablone”, “Werkzeug”, “Öl”) nachbestellt wurden.
	•	Beispiel:
	•	Ein Eintrag wie Time: 107.00 | Schablone | reorder | Order Cost: 20 zeigt, dass um 107 Zeiteinheiten eine Nachbestellung stattfand, was 20 Kosten verursacht hat.
	•	Started Jobs:
	•	Bedeutung: Eine Liste der Jobs, die in diesem Schritt gestartet wurden, mit ihrem Namen und ihrer Priorität.
	•	Beispiel:
	•	Job_26 | 7 bedeutet, Job_26 mit Priorität 7 wurde gestartet.
	•	Newly Finished Jobs:
	•	Bedeutung: Zeigt, welche Jobs in diesem Schritt abgeschlossen wurden, inklusive ihrer Fertigstellungszeit und Priorität.
	•	Beispiel:
	•	Job_26 | 118.001 | 7 zeigt, dass Job_26 um 118.001 Zeiteinheiten fertiggestellt wurde und Priorität 7 hat.



4. Performance Metriken (Nach Simulation)

Am Ende der Simulation werden zusammenfassende Leistungskennzahlen ausgegeben:
	•	Durchschnittlicher Makespan:
	•	Bedeutung: Der durchschnittlich erreichte Produktionszeitwert über mehrere Durchläufe.
	•	Beispiel: 459.90 bedeutet, im Durchschnitt dauerte die gesamte Produktion ca. 460 Zeiteinheiten.
	•	Bester Makespan:
	•	Bedeutung: Der kürzeste (beste) erreichte Makespan in den Durchläufen.
	•	Beispiel: 409.001 zeigt, dass in einem Durchlauf die Produktion in ca. 409 Zeiteinheiten abgeschlossen wurde.
	•	Schlechtester Makespan:
	•	Bedeutung: Der längste (schlechteste) erreichte Makespan in den Durchläufen.
	•	Beispiel: 510.001 bedeutet, dass in einem Durchlauf die Produktion 510 Zeiteinheiten dauerte.




Zusammenfassung
	•	Evaluations- und Trainingslogs liefern laufend Feedback darüber, wie gut der Agent lernt:
	•	Kürzere Episodenlängen und steigende Episodenbelohnungen (ep_rew_mean) deuten auf verbesserte Effizienz hin.
	•	Trainingsmetriken wie approx_kl und loss geben Hinweise darauf, wie stabil das Training verläuft.
	•	Simulationsschritte zeigen, wie sich die Produktionsumgebung verändert:
	•	Aktionen des Agenten (z. B. [2 18]) bestimmen, wie viele Jobs und welche Strategie verwendet werden.
	•	Metriken wie “Finished Jobs”, “Waiting Jobs” und “Total Cost” geben Einblick in den Fortschritt und die Kosten der Produktion.
	•	“Reorder Events” dokumentieren Nachbestellungen von Hilfsmitteln, die Kosten verursachen.
	•	Performance Metriken am Ende fassen den Gesamterfolg des Agenten zusammen (Durchschnitt, bester und schlechtester Makespan).

Diese Logs helfen dir, den Lernfortschritt und die Effizienz deines Produktionsplanungs-Systems zu überwachen – von den ersten, ineffizienten Schritten bis hin zu den optimierten Entscheidungen, die zu geringeren Produktionszeiten und Kosten führen.