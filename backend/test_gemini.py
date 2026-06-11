import json
import urllib.request
from io import StringIO
import csv
import random

ROWS = [["energy_source", "region", "co2_factor_kg_per_MJ", "loan_approved"]]
for _ in range(50):
    region = random.choice(["north", "south"])
    loan_approved = str(random.choice([0, 1]))
    ROWS.append(["coal", region, "100", loan_approved])

buf = StringIO()
writer = csv.writer(buf)
writer.writerows(ROWS)
csv_bytes = buf.getvalue().encode()

boundary = "----TestBoundary12345678"
body = (
    f"--{boundary}\r\n"
    f"Content-Disposition: form-data; name=\"target\"\r\n\r\n"
    f"loan_approved\r\n"
    f"--{boundary}\r\n"
    f"Content-Disposition: form-data; name=\"sensitive\"\r\n\r\n"
    f"region\r\n"
    f"--{boundary}\r\n"
    f"Content-Disposition: form-data; name=\"file\"; filename=\"test.csv\"\r\n"
    f"Content-Type: text/csv\r\n\r\n"
).encode() + csv_bytes + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/analyze",
    data=body,
    method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

try:
    with urllib.request.urlopen(req) as resp:
        print(json.dumps(json.loads(resp.read()), indent=2))
except Exception as e:
    print("Error:", e, e.read().decode())
