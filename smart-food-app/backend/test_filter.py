"""Quick test to verify veg/non-veg filtering works correctly."""
import requests
import json

BASE = "http://127.0.0.1:5000"
output_lines = []

def log(msg):
    output_lines.append(msg)

def test(label, url, expected_type=None):
    log(f"\n{'='*60}")
    log(f"TEST: {label}")
    log(f"URL: {url}")
    r = requests.get(url)
    data = r.json()
    
    items = data.get("results", data.get("items", []))
    types = [item.get("type") for item in items]
    detected = data.get("detected_type", "N/A")
    
    log(f"Detected type: {detected}")
    log(f"Total results: {len(items)}")
    log(f"Veg: {types.count('veg')} | Non-veg: {types.count('non-veg')}")
    log(f"Names: {[item.get('name') for item in items[:5]]}")
    
    if expected_type == "veg":
        if "non-veg" in types:
            log("[FAIL] Non-veg items found in veg-only query!")
        else:
            log("[PASS] Only veg items returned")
    elif expected_type == "non-veg":
        if "veg" in types:
            log("[FAIL] Veg items found in non-veg query!")
        else:
            log("[PASS] Only non-veg items returned")
    else:
        log(f"[INFO] Mixed results (no filter applied)")

test("Veg food search", f"{BASE}/search?q=veg+food", "veg")
test("Chicken search (auto-detect)", f"{BASE}/search?q=chicken", "non-veg")
test("Pizza with type=veg", f"{BASE}/search?q=pizza&type=veg", "veg")
test("Recommend type=veg", f"{BASE}/recommend?type=veg", "veg")
test("Paneer search (auto-detect veg)", f"{BASE}/search?q=paneer", "veg")
test("Pizza no filter (mixed OK)", f"{BASE}/search?q=pizza", None)

log(f"\n{'='*60}")
log("All tests complete.")

# Write to file in utf-8
with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
