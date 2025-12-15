import time
import gzip
import boto3
import sqlite3
import pandas as pd

from io import BytesIO
from botocore import UNSIGNED
from botocore.client import Config

def main():

    bucket_name = "openaq-data-archive"
    location_id = 3036
    start_year = 2019
    end_year = 2024

    db_path = "./database/raw/openAQ.db"
    table_name = "air_quality"

    conn = sqlite3.connect(db_path)

    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
            "location_id" BIGINT,
            "sensors_id" BIGINT,
            "location" VARCHAR,
            "datetime" TIMESTAMP,
            "lat" DOUBLE,
            "lon" DOUBLE,
            "parameter" VARCHAR,
            "units" VARCHAR,
            "value" DOUBLE,
            "month" VARCHAR,
            "year" BIGINT,
            "ingestion_datetime" TIMESTAMP
    );
    """)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    for year in range(start_year, end_year + 1):
        prefix = f"records/csv.gz/locationid={location_id}/year={year}/"

        print(f"\n--- Fetching data for year {year} ---")

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                print(f"Processing {key}")

                stream = BytesIO()
                s3.download_fileobj(bucket_name, key, stream)
                stream.seek(0)

                with gzip.GzipFile(fileobj=stream) as gz:
                    df = pd.read_csv(gz)


                df.to_sql(table_name, conn, if_exists="append", index=False)

    conn.close()

if __name__ == "__main__":
    main()