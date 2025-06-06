# hadnwash_eval

Data Guide:
## 1.
Each folder in "raw_data" is named after the label (handwash score) it contains.
## 2.
Each labeled folder contains subfolders for all the recorded sensors separately
## 3.
Each sensor csv contains all the data for 10 experiments one after the other. All experiments lasted for 40 seconds, even if I finished earler. If I finished earlier, I stopped any movement until the time ran out.
## 4.
Since the granularity of all sensors was 10 samples per second, a new experiments starts every 400 rows (so experiments start at ds[0], ds[400], ds[800] etc.)