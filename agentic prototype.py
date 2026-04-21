import pandas as pd
import re
from difflib import SequenceMatcher
from datetime import datetime
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_FILE = r"C:\Users\JOSEPH FIADZO\Desktop\Agentic_AI_Prototype_Project\taxpayer_data.csv"
OUTPUT_DIR = r"C:\Users\JOSEPH FIADZO\Desktop\Agentic_AI_Prototype_Project\outputs"

#os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
try:
    df = pd.read_csv(INPUT_FILE, dtype = str)  # Load all as string to prevent type coercion
    df = df.where(pd.notna(df), other=None)  # Normalize empty cells to None
    print(f"Loaded {len(df)} records from CSV.")
except FileNotFoundError:
    print(f"File not found: {INPUT_FILE}")
    exit()
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit()

audit_log = []
issues = []
duplicates = []

# -----------------------------
# HELPER: SAFE COMPARISON (handles None/NaN)
# -----------------------------
def safe_compare(a, b):
    a_str = "" if a is None or (isinstance(a, float) and pd.isna(a)) else str(a).strip()
    b_str = "" if b is None or (isinstance(b, float) and pd.isna(b)) else str(b).strip()
    return a_str != b_str

# -----------------------------
# CLEANING FUNCTIONS
# -----------------------------
def clean_name(name):
    """Title-case and strip whitespace. Flag names with digits."""
    if not name or pd.isna(name):
        return None, "MISSING"

    name = str(name).strip()

    if any(char.isdigit() for char in name):
        return name, "SUSPECT"  # Name contains numbers — flag it

    cleaned = " ".join(word.capitalize() for word in name.split())
    return cleaned, "OK"


def clean_phone(phone):
    """Normalize Ghanaian phone numbers to +233 format."""
    if not phone or str(phone).strip() in {"", "nan", "None"}:
        return None, "MISSING"

    phone = str(phone).strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")

    # Remove country code prefix if duplicated
    if phone.startswith("+2330"):
        phone = "+233" + phone[5:]
    elif phone.startswith("2330"):
        phone = "+233" + phone[4:]
    elif phone.startswith("0"):
        phone = "+233" + phone[1:]
    elif phone.startswith("233"):
        phone = "+" + phone
    elif phone.startswith("+233"):
        pass  # Already correct
    else:
        return phone, "INVALID"

    # Validate final length: +233 + 9 digits = 13 chars
    if len(phone) != 13 or not phone[1:].isdigit():
        return phone, "INVALID"

    return phone, "OK"


def clean_email(email):
    """Validate and lowercase email addresses."""
    if not email or str(email).strip() in {"", "nan", "None"}:
        return None, "MISSING"

    email = str(email).strip().lower()

    # Basic structural validation
    if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$", email):
        return email, "OK"

    # Has @ but malformed domain
    if "@" in email:
        return email, "SUSPECT"

    return email, "INVALID"


def clean_dob(dob):
    """Parse and standardize date of birth to YYYY-MM-DD."""
    if not dob or str(dob).strip() in {"", "nan", "None"}:
        return None, "MISSING"

    try:
        parsed = pd.to_datetime(dob, day_first=True)  # handles DD/MM/YYYY
        # Sanity check: age between 18 and 100
        age = (datetime.now() - parsed).days // 365
        if age < 18 or age > 100:
            return parsed.strftime("%Y-%m-%d"), "SUSPECT"
        return parsed.strftime("%Y-%m-%d"), "OK"
    except Exception:
        return str(dob), "INVALID"


def similarity(a, b):
    a = str(a) if a else ""
    b = str(b) if b else ""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# AGENT LOOP: CLEAN + AUDIT
cleaned_df = df.copy()

for index, row in df.iterrows():

    #NAME
    original = row.get("Full_Name")
    cleaned, status = clean_name(original)
    cleaned_df.at[index, "Full_Name"] = cleaned

    if status == "MISSING":
        issues.append({"row": index, "field": "Full_Name", "issue": "Missing value", "status": status})
    elif status == "SUSPECT":
        issues.append({"row": index, "field": "Full_Name", "issue": "Name contains digits", "status": status})

    if safe_compare(original, cleaned):
        audit_log.append({
            "row": index, "field": "Full_Name",
            "before": original, "after": cleaned,
            "reason": "Standardized name format", "status": status
        })

    #PHONE
    original = row.get("Phone_Number")
    cleaned, status = clean_phone(original)
    cleaned_df.at[index, "Phone_Number"] = cleaned

    if status in ("MISSING", "INVALID", "SUSPECT"):
        issues.append({"row": index, "field": "Phone_Number", "issue": f"Phone {status.lower()}", "status": status})

    if safe_compare(original, cleaned):
        audit_log.append({
            "row": index, "field": "Phone_Number",
            "before": original, "after": cleaned,
            "reason": "Standardized phone to +233 format", "status": status
        })

    #EMAIL
    original = row.get("Email")
    cleaned, status = clean_email(original)
    cleaned_df.at[index, "Email"] = cleaned

    if status in ("MISSING", "INVALID", "SUSPECT"):
        issues.append({"row": index, "field": "Email", "issue": f"Email {status.lower()}", "status": status})

    if safe_compare(original, cleaned):
        audit_log.append({
            "row": index, "field": "Email",
            "before": original, "after": cleaned,
            "reason": "Validated and lowercased email", "status": status
        })

    #DATE OF BIRTH
    original = row.get("Date_of_Birth")
    cleaned, status = clean_dob(original)
    cleaned_df.at[index, "Date_of_Birth"] = cleaned

    if status in ("MISSING", "INVALID", "SUSPECT"):
        issues.append({"row": index, "field": "Date_of_Birth", "issue": f"DOB {status.lower()}", "status": status})

    if safe_compare(original, cleaned):
        audit_log.append({
            "row": index, "field": "Date_of_Birth",
            "before": original, "after": cleaned,
            "reason": "Standardized date to YYYY-MM-DD", "status": status
        })


# DUPLICATE DETECTION (vectorized phone + fuzzy name)

# Step 1: Exact phone duplicates (fast)
phone_dupes = cleaned_df[
    cleaned_df["Phone_Number"].notna() &
    ~cleaned_df["Phone_Number"].isin(["INVALID", "MISSING"])
].duplicated(subset=["Phone_Number"], keep=False)

exact_phone_dup_indices = cleaned_df[phone_dupes].index.tolist()

for i in range(len(exact_phone_dup_indices)):
    for j in range(i + 1, len(exact_phone_dup_indices)):
        idx_i = exact_phone_dup_indices[i]
        idx_j = exact_phone_dup_indices[j]
        if cleaned_df.at[idx_i, "Phone_Number"] == cleaned_df.at[idx_j, "Phone_Number"]:
            duplicates.append({
                "record_1": idx_i,
                "record_2": idx_j,
                "reason": "Exact phone match",
                "confidence": "high"
            })

# Step 2: Fuzzy name similarity (only on records without exact phone dupes)
non_dup_indices = [i for i in cleaned_df.index if i not in exact_phone_dup_indices]

for i in range(len(non_dup_indices)):
    for j in range(i + 1, len(non_dup_indices)):
        idx_i = non_dup_indices[i]
        idx_j = non_dup_indices[j]   
        name_sim = similarity(
            cleaned_df.at[idx_i, "Full_Name"],
            cleaned_df.at[idx_j, "Full_Name"]
        )
        if name_sim > 0.85:
            duplicates.append({
                "record_1": idx_i,  
                "record_2": idx_j, 
                "reason": f"Similar name (score: {name_sim:.2f})",
                "confidence": "medium"
            })


# -----------------------------
# PIPELINE DECISION GATE
# -----------------------------
def pipeline_decision(issues, duplicates, total):
    issue_rate = len(issues) / max(total, 1)
    if issue_rate > 0.5:
        return "STOP", f"Over 50% of records have issues ({len(issues)}/{total})"
    if len(duplicates) > 50:
        return "REVIEW", f"High duplicate count: {len(duplicates)}"
    return "PROCEED", "Data quality acceptable"

decision, reason = pipeline_decision(issues, duplicates, len(df))

# -----------------------------
# SAVE OUTPUTS
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

cleaned_df.to_csv(f"{OUTPUT_DIR}\\cleaned_data_{timestamp}.csv", index=False)
pd.DataFrame(audit_log).to_csv(f"{OUTPUT_DIR}\\audit_log_{timestamp}.csv", index=False)
pd.DataFrame(issues).to_csv(f"{OUTPUT_DIR}\\issues_{timestamp}.csv", index=False)
pd.DataFrame(duplicates).to_csv(f"{OUTPUT_DIR}\\duplicates_{timestamp}.csv", index=False)


# -----------------------------
# SUMMARY REPORT
# -----------------------------
print("\n" + "=" * 45)
print("AGENTIC AI PIPELINE REPORT")
print("=" * 45)
print(f"  Total Records Processed : {len(df)}")
print(f"  Audit Changes Logged    : {len(audit_log)}")
print(f"  Issues Detected         : {len(issues)}")
print(f"    ↳ Missing values      : {sum(1 for i in issues if i['status'] == 'MISSING')}")
print(f"    ↳ Invalid values      : {sum(1 for i in issues if i['status'] == 'INVALID')}")
print(f"    ↳ Suspect values      : {sum(1 for i in issues if i['status'] == 'SUSPECT')}")
print(f"  Duplicates Flagged      : {len(duplicates)}")
print(f"    ↳ High confidence     : {sum(1 for d in duplicates if d['confidence'] == 'high')}")
print(f"    ↳ Medium confidence   : {sum(1 for d in duplicates if d['confidence'] == 'medium')}")
print(f"  Pipeline Decision       : {decision}")
print(f"  Reason                  : {reason}")
print(f"  Outputs saved to        : {OUTPUT_DIR}")
print(f"  Timestamp               : {timestamp}")
print("=" * 45)
print(f"  Status: {' COMPLETE' if decision != 'STOP' else ' STOPPED - Review issues before proceeding'}")
print("=" * 45)       