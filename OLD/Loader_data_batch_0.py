import json
from collections import defaultdict

# The existing dictionaries
openalex_field_to_domain = {
    'Biochemistry, Genetics and Molecular Biology': 'Life Sciences',
    'Engineering': 'Physical Sciences',
    'Environmental Science': 'Physical Sciences',
    'Mathematics': 'Physical Sciences',
    'Social Sciences': 'Social Sciences',
    'Physics and Astronomy': 'Physical Sciences',
    'Economics, Econometrics and Finance': 'Social Sciences',
    'Arts and Humanities': 'Social Sciences',
    'Chemistry': 'Physical Sciences',
    'Agricultural and Biological Sciences': 'Life Sciences',
    'Medicine': 'Health Sciences',
    'Computer Science': 'Physical Sciences',
    'Psychology': 'Social Sciences',
    'Chemical Engineering': 'Physical Sciences',
    'Nursing': 'Health Sciences',
    'Pharmacology, Toxicology and Pharmaceutics': 'Life Sciences',
    'Business, Management and Accounting': 'Social Sciences',
    'Neuroscience': 'Life Sciences',
    'Materials Science': 'Physical Sciences',
    'Health Professions': 'Health Sciences',
    'Immunology and Microbiology': 'Life Sciences',
    'Earth and Planetary Sciences': 'Physical Sciences',
    'Energy': 'Physical Sciences',
    'Dentistry': 'Health Sciences',
    'Veterinary': 'Health Sciences',
    'Decision Sciences': 'Social Sciences',
    'No Field': 'No Domain'
}

# Create a domain to field mapping
domain_to_field = defaultdict(list)

for field, domain in openalex_field_to_domain.items():
    domain_to_field[domain].append(field)

# Convert defaultdict to regular dict for JSON serialization
domain_to_field = dict(domain_to_field)

# Write to JSON file
with open('domain_to_field.json', 'w') as f:
    json.dump(domain_to_field, f, indent=2)

print("JSON file 'domain_to_field.json' has been created.")

# Print the content of the JSON file
print("\nContent of domain_to_field.json:")
print(json.dumps(domain_to_field, indent=2))



{
  "Life Sciences": [
    "Biochemistry, Genetics and Molecular Biology",
    "Agricultural and Biological Sciences",
    "Pharmacology, Toxicology and Pharmaceutics",
    "Neuroscience",
    "Immunology and Microbiology"
  ],
  "Physical Sciences": [
    "Engineering",
    "Environmental Science",
    "Mathematics",
    "Physics and Astronomy",
    "Chemistry",
    "Computer Science",
    "Chemical Engineering",
    "Materials Science",
    "Earth and Planetary Sciences",
    "Energy"
  ],
  "Social Sciences": [
    "Social Sciences",
    "Economics, Econometrics and Finance",
    "Arts and Humanities",
    "Psychology",
    "Business, Management and Accounting",
    "Decision Sciences"
  ],
  "Health Sciences": [
    "Medicine",
    "Nursing",
    "Health Professions",
    "Dentistry",
    "Veterinary"
  ],
  "No Domain": [
    "No Field"
  ]
}