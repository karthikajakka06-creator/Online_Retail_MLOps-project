from pathlib import Path
import pandas as pd, joblib, json, numpy as np
ROOT=Path(__file__).resolve().parent
m=joblib.load(ROOT/"models/online_retail_model.joblib")
X=pd.read_csv(ROOT/"data/processed/test_features.csv")
pred=m.predict(X)
report={"rows_monitored":len(X),"predicted_high_value":int((pred==1).sum()),"predicted_regular_value":int((pred==0).sum()),"high_value_rate":float(np.mean(pred))}
(ROOT/"reports/monitoring_report.json").write_text(json.dumps(report,indent=2))
print("[LAB 12] Prediction monitoring report",report)
