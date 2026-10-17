import io
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

CUR = "$"   # currency symbol used in the report


def _clean(text):
    """fpdf core fonts only support basic characters."""
    text = str(text).replace("**", "").replace("*", "")
    return text.encode("latin-1", "replace").decode("latin-1")


def compute_stats(d):
    """d = one latest prediction per customer."""
    by_contract = d.groupby("contract")["risk_score"].mean()
    return {
        "customers": len(d),
        "high": int((d["risk_level"] == "High").sum()),
        "medium": int((d["risk_level"] == "Medium").sum()),
        "low": int((d["risk_level"] == "Low").sum()),
        "avg_risk": float(d["risk_score"].mean()),
        "rev_risk": float((d["risk_score"] / 100 * d["monthly"]).sum()),
        "high_rev": float(d.loc[d["risk_level"] == "High", "monthly"].sum()),
        "worst_contract": by_contract.idxmax(),
    }


def _chart(s):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.bar(["Low", "Medium", "High"], [s["low"], s["medium"], s["high"]],
        color=["#16A34A", "#D97706", "#DC2626"])
    ax.set_ylabel("Customers")
    ax.set_title("Customers by risk level")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf(d, summary):
    s = compute_stats(d)
    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 8)
    pdf.cell(0, 8, "ChurnGuard - Executive Churn Audit Report",
            new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(10)
    pdf.cell(0, 6, "Generated on " + datetime.now().strftime("%d %b %Y, %H:%M"),
            new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(15, 23, 42)
    pdf.set_y(40)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Key figures", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    rows = [("Customers analysed", s["customers"]),
            ("High risk customers", s["high"]),
            ("Medium risk customers", s["medium"]),
            ("Low risk customers", s["low"]),
            ("Average churn risk", f"{s['avg_risk']:.1f}%"),
            ("Expected monthly revenue at risk", f"{CUR}{s['rev_risk']:,.0f}"),
            ("Monthly bill of high-risk customers", f"{CUR}{s['high_rev']:,.0f}")]
    for label, value in rows:
        pdf.cell(110, 7, label, border=1)
        pdf.cell(60, 7, str(value), border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Executive summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _clean(summary), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf.image(_chart(s), x=40, w=120)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "3. High-risk queue (top 15)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 10)
    widths = [45, 35, 30, 30, 30]
    for w, h in zip(widths, ["Customer", "Contract", "Tenure", "Monthly", "Risk %"]):
        pdf.cell(w, 7, h, border=1, fill=False)
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 10)
    queue = (d[d["risk_level"] == "High"]
            .sort_values("risk_score", ascending=False).head(15))
    for _, r in queue.iterrows():
        vals = [str(r["customer"])[:20], str(r["contract"]), str(r["tenure"]),
                f"{CUR}{r['monthly']:.0f}", f"{r['risk_score']:.1f}"]
        for w, v in zip(widths, vals):
            pdf.cell(w, 7, _clean(v), border=1)
        pdf.ln(7)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "4. Recommended retention actions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for tip in ["Offer 1-year or 2-year plan discounts to month-to-month customers.",
                "Give free Tech Support trials to high-risk customers without it.",
                "Encourage automatic payment with a small bill credit.",
                "Contact the high-risk queue within 7 days."]:
        pdf.multi_cell(0, 6, "- " + tip, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Method: XGBoost churn model with SHAP explainability. "
                "Revenue at risk = churn probability x monthly bill, summed "
                "over customers. Based on the latest prediction per customer.",
                new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())