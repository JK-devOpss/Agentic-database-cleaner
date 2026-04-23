from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import re
from difflib import SequenceMatcher
from datetime import datetime
import io
import json

app = FastAPI(title="Agentic Data Cleaner API")

# ----------------------------- 
# CORS — allow Lovable + any frontend
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your Lovable URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# HELPER
# -----------------------------
def safe_compare(a, b):
    a_str = "" if a is None or (isinstance(a, float) and pd.isna(a)) else str(a).strip()
    b_str = "" if b is None or (isinstance(b, float) and pd.isna(b)) else str(b).strip()
    return a_str != b_str


# -----------------------------
# CLEANING FUNCTIONS
# -----------------------------
def clean_name(name):
    if not name or pd.isna(name):
        return None, "MISSING"
    name = str(name).strip()
    if any(char.isdigit() for char in name):
        return name, "SUSPECT"
    cleaned = " ".join(word.capitalize() for word in name.split())
    return cleaned, "OK"


def clean_phone(phone):
    if not phone or str(phone).strip() in {"", "nan", "None"}:
        return None, "MISSING"
    phone = str(phone).strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if phone.startswith("+2330"):
        phone = "+233" + phone[5:]
    elif phone.startswith("2330"):
        phone = "+233" + phone[4:]
    elif phone.startswith("0"):
        phone = "+233" + phone[1:]
    elif phone.startswith("233"):
        phone = "+" + phone
    elif phone.startswith("+233"):
        pass
    else:
        return phone, "INVALID"
    if len(phone) != 13 or not phone[1:].isdigit():
        return phone, "INVALID"
    return phone, "OK"


def clean_email(email):
    if not email or str(email).strip() in {"", "nan", "None"}:
        return None, "MISSING"
    email = str(email).strip().lower()
    if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$", email):
        return email, "OK"
    if "@" in email:
        return email, "SUSPECT"
    return email, "INVALID"


def clean_dob(dob):
    if not dob or str(dob).strip() in {"", "nan", "None"}:
        return None, "MISSING"
    try:
        parsed = pd.to_datetime(dob, day_first=False, format= "mixed")
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


def pipeline_decision(issues, duplicates, total):
    issue_rate = len(issues) / max(total, 1)
    if issue_rate > 0.5:
        return "STOP", f"Over 50% of records have issues ({len(issues)}/{total})"
    if len(duplicates) > 50:
        return "REVIEW", f"High duplicate count: {len(duplicates)}"
    return "PROCEED", "Data quality acceptable"


# -----------------------------
# CORE CLEANING PIPELINE
# -----------------------------
def run_pipeline(df: pd.DataFrame):
    df = df.where(pd.notna(df), other=None)
    df = df.fillna("")
    
    audit_log = []
    issues = []
    duplicates = []
    cleaned_df = df.copy()

    for index, row in df.iterrows():
        # NAME
        original = row.get("Full_Name")
        cleaned, status = clean_name(original)
        cleaned_df.at[index, "Full_Name"] = cleaned
        if status in ("MISSING", "SUSPECT"):
            issues.append({"row": index, "field": "Full_Name", "issue": "Missing or suspect name", "status": status})
        if safe_compare(original, cleaned):
            audit_log.append({"row": index, "field": "Full_Name", "before": original, "after": cleaned, "reason": "Standardized name format", "status": status})

        # PHONE
        original = row.get("Phone_Number")
        cleaned, status = clean_phone(original)
        cleaned_df.at[index, "Phone_Number"] = cleaned
        if status in ("MISSING", "INVALID", "SUSPECT"):
            issues.append({"row": index, "field": "Phone_Number", "issue": f"Phone {status.lower()}", "status": status})
        if safe_compare(original, cleaned):
            audit_log.append({"row": index, "field": "Phone_Number", "before": original, "after": cleaned, "reason": "Standardized to +233 format", "status": status})

        # EMAIL
        original = row.get("Email")
        cleaned, status = clean_email(original)
        cleaned_df.at[index, "Email"] = cleaned
        if status in ("MISSING", "INVALID", "SUSPECT"):
            issues.append({"row": index, "field": "Email", "issue": f"Email {status.lower()}", "status": status})
        if safe_compare(original, cleaned):
            audit_log.append({"row": index, "field": "Email", "before": original, "after": cleaned, "reason": "Validated and lowercased email", "status": status})

        # DOB
        original = row.get("Date_of_Birth")
        cleaned, status = clean_dob(original)
        cleaned_df.at[index, "Date_of_Birth"] = cleaned
        if status in ("MISSING", "INVALID", "SUSPECT"):
            issues.append({"row": index, "field": "Date_of_Birth", "issue": f"DOB {status.lower()}", "status": status})
        if safe_compare(original, cleaned):
            audit_log.append({"row": index, "field": "Date_of_Birth", "before": original, "after": cleaned, "reason": "Standardized date to YYYY-MM-DD", "status": status})

    # DUPLICATE DETECTION
    valid_phones_mask = (
        cleaned_df["Phone_Number"].notna() &
        ~cleaned_df["Phone_Number"].isin(["INVALID", "MISSING"])
    )
    phone_dupes = cleaned_df[valid_phones_mask].duplicated(subset=["Phone_Number"], keep=False)
    phone_dupes_reindexed = phone_dupes.reindex(cleaned_df.index, fill_value=False)
    exact_phone_dup_indices = cleaned_df[phone_dupes_reindexed].index.tolist()

    for i in range(len(exact_phone_dup_indices)):
        for j in range(i + 1, len(exact_phone_dup_indices)):
            idx_i = exact_phone_dup_indices[i]
            idx_j = exact_phone_dup_indices[j]
            if cleaned_df.at[idx_i, "Phone_Number"] == cleaned_df.at[idx_j, "Phone_Number"]:
                duplicates.append({
                    "record_1": int(idx_i), 
                    "record_2": int(idx_j), 
                    "reason": "Exact phone match", 
                    "confidence": "high"
                })

    non_dup_indices = [i for i in cleaned_df.index if i not in exact_phone_dup_indices]
    for i in range(len(non_dup_indices)):
        for j in range(i + 1, len(non_dup_indices)):
            idx_i = non_dup_indices[i]
            idx_j = non_dup_indices[j]
            name_sim = similarity(cleaned_df.at[idx_i, "Full_Name"], cleaned_df.at[idx_j, "Full_Name"])
            if name_sim > 0.85:
                duplicates.append({"record_1": int(idx_i), "record_2": int(idx_j), "reason": f"Similar name (score: {name_sim:.2f})", "confidence": "medium"})

    decision, reason = pipeline_decision(issues, duplicates, len(df))
    cleaned_df = cleaned_df.fillna("").replace([float('inf'), float('-inf')], "")
    return cleaned_df, audit_log, issues, duplicates, decision, reason


# -----------------------------
# ROUTES
# -----------------------------

@app.get("/")
def root():
    return {"message": "Agentic Data Cleaner API is running ✅"}


@app.post("/clean")
async def clean_data(file: UploadFile = File(...)):
    """
    Upload a CSV file. Returns a JSON report with summary,
    issues, duplicates, audit log, and cleaned data preview.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), dtype=str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    cleaned_df, audit_log, issues, duplicates, decision, reason = run_pipeline(df)

    return {
        "summary": {
            "total_records": len(df),
            "audit_changes": len(audit_log),
            "total_issues": len(issues),
            "missing": sum(i["status"] == "MISSING" for i in issues),
            "invalid": sum(i["status"] == "INVALID" for i in issues),
            "suspect": sum(i["status"] == "SUSPECT" for i in issues),
            "duplicates_flagged": len(duplicates),
            "high_confidence_dupes": sum(d["confidence"] == "high" for d in duplicates),
            "medium_confidence_dupes": sum(d["confidence"] == "medium" for d in duplicates),
            "pipeline_decision": decision,
            "reason": reason,
        },
        "issues": issues[:100],           # cap at 100 for response size
        "duplicates": duplicates[:100],
        "audit_log": audit_log[:100],
        "preview": cleaned_df.head(20).to_dict(orient="records"),  # first 20 cleaned rows
    }


@app.post("/clean/download")
async def clean_and_download(file: UploadFile = File(...)):
    """
    Upload a CSV. Returns the fully cleaned CSV as a downloadable file.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), dtype=str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    cleaned_df, _, _, _, _, _ = run_pipeline(df)

    output = io.StringIO()
    cleaned_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"}
    )