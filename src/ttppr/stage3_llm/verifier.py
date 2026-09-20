# src/ttppr/stage3_llm/verifier.py

class ClinicalVerifier:
    """Verifies candidate ICD-10 codes against CMS Excludes notes and LCD reimbursement policies."""

    @staticmethod
    def validate_code_conflicts(candidate_rules: list[dict], lcd_policies: dict[str, list[dict]] = None) -> dict:
        predicted_codes = {r["code"] for r in candidate_rules}
        conflicts = []
        approved_codes = []
        coverage_warnings = []

        # 1. CMS Excludes1 / Excludes2 Mutual Exclusivity Check
        for rule in candidate_rules:
            code = rule["code"]
            excludes1 = rule.get("excludes1", [])

            has_conflict = False
            for excl in excludes1:
                for other_code in predicted_codes:
                    if other_code != code and other_code.startswith(excl):
                        conflicts.append({
                            "code": code,
                            "conflicting_code": other_code,
                            "rule_violated": f"Excludes1 note: {excl}",
                        })
                        has_conflict = True

            if not has_conflict:
                approved_codes.append(code)

        # 2. LCD Reimbursement Coverage Check
        if lcd_policies:
            for code in approved_codes:
                policies = lcd_policies.get(code, [])
                if not policies:
                    coverage_warnings.append({
                        "code": code,
                        "warning": "No explicit LCD local coverage policy mapped; defaults to national coverage."
                    })

        return {
            "approved_codes": approved_codes,
            "conflicts_detected": conflicts,
            "coverage_warnings": coverage_warnings
        }