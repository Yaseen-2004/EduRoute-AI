
import json
from pathlib import Path

# Resolve project directories
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "config"
BRANCHES_DIR = CONFIG_DIR / "branches"

DISCIPLINES_FILE = CONFIG_DIR / "disciplines.json"
COMMON_SUBJECTS_FILE = CONFIG_DIR / "common_subjects.json"


def read_json(file_path):
    """Read and return JSON data."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_subjects(subjects, catalogue_name):
    """Validate subject records and their required fields."""

    if not isinstance(subjects, list):
        raise ValueError(f"{catalogue_name}: subjects must be a list")

    subject_ids = []

    for subject in subjects:
        required_fields = ["id", "name", "year", "topics"]

        for field in required_fields:
            if field not in subject:
                raise ValueError(
                    f"{catalogue_name}: missing field '{field}'"
                )

        if not isinstance(subject["topics"], list) or not subject["topics"]:
            raise ValueError(
                f"{catalogue_name}: invalid topics for {subject['id']}"
            )

        if subject["year"] not in [1, 2, 3, 4]:
            raise ValueError(
                f"{catalogue_name}: invalid academic year for {subject['id']}"
            )

        subject_ids.append(subject["id"])

    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError(f"{catalogue_name}: duplicate subject IDs found")

    return True


def load_catalogue():
    """Load common and available specialized B.Tech catalogues."""

    discipline_data = read_json(DISCIPLINES_FILE)
    common_data = read_json(COMMON_SUBJECTS_FILE)

    catalogue = {
        "metadata": discipline_data,
        "groups": {},
        "subjects": [],
        "missing_branches": []
    }

    # Load common engineering subjects
    common_subjects = common_data["subjects"]

    validate_subjects(common_subjects, "COMMON")

    catalogue["groups"]["COMMON"] = common_subjects
    catalogue["subjects"].extend(common_subjects)

    # Load specialized branch catalogues
    for branch in discipline_data["groups"]:

        if branch["category"] != "specialized":
            continue

        branch_code = branch["code"]
        branch_file = BRANCHES_DIR / f"{branch_code.lower()}.json"

        if not branch_file.exists():
            catalogue["missing_branches"].append(branch_code)
            continue

        branch_data = read_json(branch_file)
        branch_subjects = branch_data["subjects"]

        if branch_data["branch_code"] != branch_code:
            raise ValueError(
                f"Branch code mismatch in {branch_file.name}"
            )

        validate_subjects(branch_subjects, branch_code)

        expected_count = branch["planned_subject_count"]

        if len(branch_subjects) != expected_count:
            raise ValueError(
                f"{branch_code}: expected {expected_count} subjects, "
                f"found {len(branch_subjects)}"
            )

        catalogue["groups"][branch_code] = branch_subjects
        catalogue["subjects"].extend(branch_subjects)

    # Check for duplicate IDs across all loaded groups
    all_ids = [subject["id"] for subject in catalogue["subjects"]]

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate subject IDs across catalogues")

    return catalogue


def get_subjects_by_branch(catalogue, branch_code):
    """Return subjects belonging to a selected branch."""
    return catalogue["groups"].get(branch_code.upper(), [])


def search_subject(catalogue, keyword):
    """Search subject names and topics."""
    keyword = keyword.lower().strip()

    results = []

    for subject in catalogue["subjects"]:
        searchable_text = (
            subject["name"] + " " + " ".join(subject["topics"])
        ).lower()

        if keyword in searchable_text:
            results.append(subject)

    return results


if __name__ == "__main__":

    data = load_catalogue()

    print("\n========== EDUROUTE AI CATALOGUE ==========")

    print("Loaded groups:", len(data["groups"]))
    print("Loaded subjects:", len(data["subjects"]))

    print("\nLoaded disciplines:")

    for branch_code, subjects in data["groups"].items():
        print(f"{branch_code}: {len(subjects)} subjects")

    print("\nMissing branch catalogues:")
    print(data["missing_branches"])

    print("\nCATALOGUE LOADER READY")
