import json
import random
from typing import List, Dict, Tuple

# Seed for reproducibility
random.seed(42)

def generate_credit_card() -> str:
    """Generate credit card in spoken form."""
    patterns = [
        # All digits
        lambda: " ".join([str(random.randint(0, 9)) for _ in range(16)]),
        # Grouped by 4
        lambda: " ".join([" ".join([str(random.randint(0, 9)) for _ in range(4)]) for _ in range(4)]),
        # Mix of spoken and numeric
        lambda: f"{random.randint(4000, 5999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}",
        # Common test cards
        lambda: random.choice([
            "4242 4242 4242 4242",
            "5555 5555 5555 4444",
            "four two four two 4242 4242 4242",
        ])
    ]
    return random.choice(patterns)()


def generate_phone() -> str:
    """Generate phone number - simplified for better recognition."""
    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    
    # Generate 10 digits
    digits = [random.randint(0, 9) for _ in range(10)]
    
    # More consistent patterns (70% spoken, 30% numeric)
    if random.random() < 0.7:
        # All spoken - most common in STT
        return " ".join([digit_words[d] for d in digits])
    else:
        # Numeric - grouped format
        return "".join([str(d) for d in digits])


def generate_email() -> str:
    """Generate email in spoken form - NO overlapping person names."""
    first_names = ["john", "sarah", "rahul", "priya", "mike", "alice", "david", "emily", 
                   "raj", "ananya", "kumar", "sharma", "robert", "jennifer"]
    last_names = ["smith", "johnson", "kumar", "sharma", "patel", "jones", "brown", 
                  "davis", "wilson", "verma", "gupta", "singh"]
    domains = ["gmail", "outlook", "yahoo", "company", "email", "hotmail"]
    tlds = ["com", "org", "net", "in"]
    
    fname = random.choice(first_names)
    lname = random.choice(last_names)
    domain = random.choice(domains)
    tld = random.choice(tlds)
    
    # Use formats that DON'T create person name confusion
    # Use concatenated or single name only
    formats = [
        f"{fname}{lname} at {domain} dot {tld}",
        f"{fname} at {domain} dot {tld}",
        f"{fname} {random.randint(10, 99)} at {domain} dot {tld}",
        f"{fname} underscore {random.randint(100, 999)} at {domain} dot {tld}",
    ]
    
    return random.choice(formats)


def generate_person_name() -> str:
    """Generate person name."""
    first_names = ["john smith", "sarah johnson", "rahul kumar", "priya sharma", 
                   "mike jones", "alice cooper", "david wilson", "emily brown",
                   "raj patel", "ananya gupta", "ramesh verma", "deepak singh"]
    return random.choice(first_names)


def generate_date() -> str:
    """Generate date in spoken form."""
    patterns = [
        lambda: f"{random.randint(1, 31):02d} {random.randint(1, 12):02d} {random.randint(2020, 2025)}",
        lambda: f"{random.choice(['january', 'february', 'march', 'april', 'may', 'june', 'july'])} {random.randint(1, 31)} {random.randint(2020, 2025)}",
        lambda: f"first of {random.choice(['january', 'february', 'march'])} {random.randint(2020, 2025)}",
        lambda: f"{random.randint(1, 31)} {random.randint(1, 12)} {random.randint(2020, 2025)}",
    ]
    return random.choice(patterns)()


def generate_city() -> str:
    """Generate city name."""
    cities = ["mumbai", "delhi", "bangalore", "chennai", "hyderabad", "pune",
              "new york", "london", "paris", "tokyo", "sydney", "singapore",
              "boston", "san francisco", "seattle", "austin", "chicago"]
    return random.choice(cities)


def generate_location() -> str:
    """Generate location."""
    locations = ["central park", "times square", "india gate", "marine drive",
                 "mg road", "brigade road", "connaught place", "park street",
                 "manhattan", "brooklyn", "queens", "airport", "railway station"]
    return random.choice(locations)


# ============================================================================
# TEMPLATE PATTERNS (STT-style speech)
# ============================================================================

TEMPLATES = [
    # Credit card patterns
    "my credit card number is {CREDIT_CARD}",
    "card number is {CREDIT_CARD} and email is {EMAIL}",
    "my card is {CREDIT_CARD}",
    "i want to use card {CREDIT_CARD}",
    "payment with {CREDIT_CARD}",
    
    # Phone patterns (INCREASED - was 6, now 12)
    "call me on {PHONE}",
    "my number is {PHONE}",
    "you can reach me at {PHONE}",
    "phone number {PHONE}",
    "my mobile is {PHONE} and i live in {CITY}",
    "contact me on {PHONE}",
    "please call {PHONE}",
    "my phone is {PHONE}",
    "reach me at {PHONE}",
    "contact number is {PHONE}",
    "call back on {PHONE}",
    "my contact is {PHONE}",
    
    # Email patterns (INCREASED - was 5, now 12)
    "my email is {EMAIL}",
    "email id is {EMAIL}",
    "send it to {EMAIL}",
    "you can email me at {EMAIL}",
    "my email address is {EMAIL} and phone is {PHONE}",
    "email me at {EMAIL}",
    "my id is {EMAIL}",
    "send to {EMAIL}",
    "contact email {EMAIL}",
    "write to {EMAIL}",
    "reach me at {EMAIL}",
    "email address is {EMAIL}",
    
    # Person name patterns
    "my name is {PERSON_NAME}",
    "i am {PERSON_NAME} from {CITY}",
    "this is {PERSON_NAME} calling",
    "my name is {PERSON_NAME} and my phone is {PHONE}",
    
    # Date patterns
    "my appointment is on {DATE}",
    "i will travel on {DATE}",
    "meeting on {DATE}",
    "born on {DATE}",
    "visit scheduled for {DATE}",
    "i am coming on {DATE} to {CITY}",
    
    # Combined patterns (more realistic)
    "hi my name is {PERSON_NAME} and my phone number is {PHONE}",
    "i live in {CITY} and my email is {EMAIL}",
    "my card {CREDIT_CARD} was used in {CITY}",
    "i am {PERSON_NAME} from {CITY} and you can call me on {PHONE}",
    "my email is {EMAIL} and i will visit on {DATE}",
    "appointment on {DATE} for {PERSON_NAME} at {LOCATION}",
    "i am calling from {CITY} my number is {PHONE}",
    "{PERSON_NAME} will arrive on {DATE} in {CITY}",
    "my details are {PERSON_NAME} phone {PHONE} email {EMAIL}",
    
    # Complex patterns
    "i need to update my email {EMAIL} for {PERSON_NAME}",
    "my name is {PERSON_NAME} i am calling from {CITY} on {PHONE} about my appointment on {DATE}",
    "contact {PERSON_NAME} at {EMAIL} or {PHONE} in {CITY}",
]


# ============================================================================
# UTTERANCE GENERATION
# ============================================================================

def generate_utterance(template: str, idx: int) -> Dict:
    """Generate a single utterance from a template."""
    text = template
    entities = []
    
    # Track entity positions
    entity_generators = {
        "CREDIT_CARD": generate_credit_card,
        "PHONE": generate_phone,
        "EMAIL": generate_email,
        "PERSON_NAME": generate_person_name,
        "DATE": generate_date,
        "CITY": generate_city,
        "LOCATION": generate_location,
    }
    
    # Find and replace placeholders
    for entity_type, generator in entity_generators.items():
        placeholder = f"{{{entity_type}}}"
        while placeholder in text:
            # Generate entity
            entity_text = generator()
            
            # Find position
            start = text.index(placeholder)
            
            # Replace placeholder
            text = text.replace(placeholder, entity_text, 1)
            
            # Record entity
            end = start + len(entity_text)
            entities.append({
                "start": start,
                "end": end,
                "label": entity_type
            })
    
    # Sort entities by start position
    entities.sort(key=lambda x: x["start"])
    
    return {
        "id": f"utt_{idx:04d}",
        "text": text,
        "entities": entities
    }


def generate_dataset(num_samples: int, start_idx: int = 0) -> List[Dict]:
    """Generate a dataset of utterances."""
    dataset = []
    
    for i in range(num_samples):
        template = random.choice(TEMPLATES)
        utterance = generate_utterance(template, start_idx + i)
        dataset.append(utterance)
    
    return dataset


def save_dataset(dataset: List[Dict], filepath: str):
    """Save dataset to JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(dataset)} samples to {filepath}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic PII NER training data")
    parser.add_argument("--num_train", type=int, default=500, 
                       help="Number of training samples (default: 500)")
    parser.add_argument("--num_dev", type=int, default=100, 
                       help="Number of dev samples (default: 100)")
    parser.add_argument("--num_test", type=int, default=50, 
                       help="Number of test samples (default: 50)")
    args = parser.parse_args()
    
    print("🎯 Generating PII NER Dataset for IIT Madras Assignment")
    print("=" * 60)
    
    # Use SAME random seed for consistency (better generalization)
    random.seed(42)
    
    # Generate datasets
    print(f"\    Generating {args.num_train} training samples...")
    train_data = generate_dataset(args.num_train, start_idx=1)
    save_dataset(train_data, "data/train.jsonl")
    
    print(f"\    Generating {args.num_dev} dev samples...")
    dev_data = generate_dataset(args.num_dev, start_idx=args.num_train + 1)
    save_dataset(dev_data, "data/dev.jsonl")
    
    print(f"\    Generating {args.num_test} test samples...")
    test_data = generate_dataset(args.num_test, start_idx=args.num_train + args.num_dev + 1)
    # Remove entities from test data (as per assignment)
    for item in test_data:
        item.pop("entities", None)
    save_dataset(test_data, "data/test.jsonl")
    
    # Statistics
    print("\ Dataset Statistics:")
    print("=" * 60)
    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples:   {len(dev_data)}")
    print(f"Test samples:  {len(test_data)}")
    
    # Entity distribution in training data
    entity_counts = {}
    for item in train_data:
        for ent in item["entities"]:
            label = ent["label"]
            entity_counts[label] = entity_counts.get(label, 0) + 1
    
    print("\n📈 Entity Distribution (Training Set):")
    for label in sorted(entity_counts.keys()):
        print(f"  {label:15s}: {entity_counts[label]:4d}")
    
    print("\n   Data generation complete!")
    print("\    Next steps:")
    print("  1. Inspect the data: head data/train.jsonl")
    print("  2. Train the model: python3 src/train.py --train data/train.jsonl --dev data/dev.jsonl")
    print("  3. Evaluate: python3 src/eval_span_f1.py --gold data/dev.jsonl --pred out/best_model/dev_pred.json")

