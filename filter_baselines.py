import pandas as pd
import sys

def main():
    cluster_app = "cluster_app_metrics.csv"
    cluster_logs = "cluster_incident_logs.csv"
    container_metrics = "container_metrics.csv"
    incident_traces = "incident_traces.csv"

    baseline_app = "baseline_app_metrics.csv"
    baseline_container = "baseline_container_metrics.csv"

    unique_ids = set()

    # extract from cluster_app (column tc)
    try:
        df1 = pd.read_csv(cluster_app, usecols=['tc'])
        unique_ids.update(df1['tc'].dropna().unique())
    except Exception as e:
        print(f"Error reading {cluster_app}: {e}")

    # extract from logs (column cmdb_id)
    try:
        df2 = pd.read_csv(cluster_logs, usecols=['cmdb_id'])
        unique_ids.update(df2['cmdb_id'].dropna().unique())
    except Exception as e:
        print(f"Error reading {cluster_logs}: {e}")

    # extract from container metrics
    try:
        df3 = pd.read_csv(container_metrics, usecols=['cmdb_id'])
        unique_ids.update(df3['cmdb_id'].dropna().unique())
    except Exception as e:
        print(f"Error reading {container_metrics}: {e}")

    # extract from traces
    try:
        df4 = pd.read_csv(incident_traces, usecols=['cmdb_id'])
        unique_ids.update(df4['cmdb_id'].dropna().unique())
    except Exception as e:
        print(f"Error reading {incident_traces}: {e}")

    print("Unique CMDB IDs found:")
    for cid in sorted(list(unique_ids)):
        print(f" - {cid}")
    print(f"\nTotal unique IDs: {len(unique_ids)}\n")

    # Filter baseline app metrics
    try:
        print(f"Filtering {baseline_app}...")
        df_b_app = pd.read_csv(baseline_app)
        orig_len = len(df_b_app)
        df_b_app = df_b_app[df_b_app['tc'].isin(unique_ids)]
        df_b_app.to_csv(baseline_app, index=False)
        print(f"  Rows before: {orig_len}, Rows after: {len(df_b_app)}")
    except Exception as e:
        print(f"Error processing {baseline_app}: {e}")

    # Filter baseline container metrics
    try:
        print(f"Filtering {baseline_container}...")
        df_b_cont = pd.read_csv(baseline_container)
        orig_len = len(df_b_cont)
        df_b_cont = df_b_cont[df_b_cont['cmdb_id'].isin(unique_ids)]
        df_b_cont.to_csv(baseline_container, index=False)
        print(f"  Rows before: {orig_len}, Rows after: {len(df_b_cont)}")
    except Exception as e:
        print(f"Error processing {baseline_container}: {e}")

if __name__ == '__main__':
    main()
