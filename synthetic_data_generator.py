import random
import csv
from datetime import datetime, timedelta
import numpy as np

# Global Constants for Customization
BASE_EMAIL_FILL_RATE = 0.30  # 30% base chance for email to be filled
JAN_JUNE_EMAIL_BOOST = 0.40  # +40% chance in Jan and June
SAFARI_BRAVE_MOBILE_EMAIL_BOOST = 0.50  # +30% chance for Safari/Brave/Mobile
FATEC_EMAIL_REDUCTION = 0.25  # -15% chance for FATEC routes

# Time spent distribution parameters
TIME_SPENT_MEAN = 180
TIME_SPENT_STD = 60
TIME_SPENT_MIN = 1
TIME_SPENT_MAX = 600

# Date range
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2027, 12, 31)

FILENAME = "./datasets/bv-web-analytics.csv"

# Data options with weights
ROUTES = {
    '/': 25,
    '/instituicoes': 15,
    '/cursos': 20,
    '/quiz': 10,
    '/vestibular/FUVEST': 8,
    '/vestibular/ENEM': 12,
    '/vestibular/FATEC': 5,
    '/vestibular/UNICAMP': 5
}

REFERRERS = {
    'chrome': 30,
    'safari': 20,
    'firefox': 15,
    'share': 8,
    'lp': 7,
    'opera': 5,
    'brave': 3,
    'duckduckgo': 3,
    'tiktok': 3,
    'ig': 2,
    'x': 2,
    'linkedin': 2
}

DEVICES = {
    'mobile': 45,
    'desktop': 35,
    'tablet': 15,
    'tv': 3,
    'vr': 2
}

EMAIL_DOMAINS = [
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'usp.br',
    'uol.com.br', 'terra.com.br', 'bol.com.br', 'ig.com.br', 'globo.com',
    'live.com', 'icloud.com', 'protonmail.com', 'edu.br', 'fatec.sp.gov.br'
]

def weighted_choice(choices_dict):
    """Select a random choice based on weights."""
    choices = list(choices_dict.keys())
    weights = list(choices_dict.values())
    return random.choices(choices, weights=weights)[0]

def generate_random_ip():
    """Generate a random IPv4 address."""
    # Generate realistic IP ranges (avoiding private/reserved ranges mostly)
    public_ranges = [
        (8, 8, 0, 0, 8, 8, 255, 255),  # Google DNS range
        (1, 1, 0, 0, 1, 1, 255, 255),  # Cloudflare
        (200, 0, 0, 0, 200, 255, 255, 255),  # Some public ranges
        (189, 0, 0, 0, 189, 255, 255, 255),  # Brazilian ISP ranges
        (177, 0, 0, 0, 177, 255, 255, 255),
    ]
    
    range_choice = random.choice(public_ranges)
    return f"{random.randint(range_choice[0], range_choice[4])}.{random.randint(range_choice[1], range_choice[5])}.{random.randint(range_choice[2], range_choice[6])}.{random.randint(range_choice[3], range_choice[7])}"

def generate_random_date():
    """Generate a random date between START_DATE and END_DATE."""
    delta = END_DATE - START_DATE
    random_days = random.randint(0, delta.days)
    return START_DATE + timedelta(days=random_days)

def generate_email(should_fill=True):
    """Generate a random email address."""
    if not should_fill:
        return ""
    
    first_names = ['ana', 'carlos', 'maria', 'joao', 'lucas', 'julia', 'pedro', 'beatriz', 'rafael', 'larissa']
    last_names = ['silva', 'santos', 'oliveira', 'souza', 'rodrigues', 'ferreira', 'alves', 'pereira', 'lima', 'gomes']
    
    first = random.choice(first_names)
    last = random.choice(last_names)
    domain = random.choice(EMAIL_DOMAINS)
    
    # Add some variation in email formats
    formats = [
        f"{first}.{last}@{domain}",
        f"{first}{last}@{domain}",
        f"{first}.{last}{random.randint(1, 99)}@{domain}",
        f"{first[0]}.{last}@{domain}"
    ]
    
    return random.choice(formats)

def calculate_email_probability(date, referrer, device, route):
    """Calculate the probability of email being filled based on biases."""
    prob = BASE_EMAIL_FILL_RATE
    
    # January and June boost
    if date.month in [1, 6]:
        prob += JAN_JUNE_EMAIL_BOOST
    
    # Safari/Brave/Mobile boost
    if referrer in ['safari', 'brave'] or device == 'mobile':
        prob += SAFARI_BRAVE_MOBILE_EMAIL_BOOST
    
    # FATEC reduction
    if '/vestibular/FATEC' in route:
        prob -= FATEC_EMAIL_REDUCTION
    
    # Ensure probability is between 0 and 1
    return max(0, min(1, prob))

def generate_time_spent():
    """Generate time spent with normal distribution."""
    time_spent = np.random.normal(TIME_SPENT_MEAN, TIME_SPENT_STD)
    return max(TIME_SPENT_MIN, min(TIME_SPENT_MAX, int(time_spent)))

def generate_synthetic_data(n_rows):
    """Generate N rows of synthetic web analytics data."""
    data = []
    
    for _ in range(n_rows):
        # Generate basic fields
        ip = generate_random_ip()
        date = generate_random_date()
        route = weighted_choice(ROUTES)
        scroll_pct = round(random.uniform(0, 100), 2)
        time_spent = generate_time_spent()
        referrer = weighted_choice(REFERRERS)
        device = weighted_choice(DEVICES)
        
        # Calculate email probability based on biases
        email_prob = calculate_email_probability(date, referrer, device, route)
        should_fill_email = random.random() < email_prob
        email = generate_email(should_fill_email)
        
        # Format date as dd/mm/yyyy
        date_str = date.strftime("%d/%m/%Y")
        
        data.append({
            'ip': ip,
            'date': date_str,
            'email': email,
            'route': route,
            'scroll_pct': scroll_pct,
            'time_spent': time_spent,
            'referrer': referrer,
            'device': device
        })
    
    return data

def save_to_csv(data, filename='synthetic_web_data.csv'):
    """Save data to CSV file."""
    fieldnames = ['ip', 'date', 'email', 'route', 'scroll_pct', 'time_spent', 'referrer', 'device']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Data saved to {filename}")

def print_sample_data(data, n_samples=5):
    """Print a sample of the generated data."""
    print(f"\nSample of {n_samples} rows:")
    print("-" * 120)
    for i, row in enumerate(data[:n_samples]):
        print(f"Row {i+1}:")
        for key, value in row.items():
            print(f"  {key}: {value}")
        print()

def analyze_biases(data):
    """Analyze the generated data to verify biases are working."""
    total_rows = len(data)
    email_filled = sum(1 for row in data if row['email'])
    
    print(f"\nBias Analysis:")
    print(f"Total rows: {total_rows}")
    print(f"Overall email fill rate: {email_filled/total_rows:.2%}")
    
    # Analyze January/June bias
    jan_june_rows = [row for row in data if row['date'].split('/')[1] in ['01', '06']]
    jan_june_filled = sum(1 for row in jan_june_rows if row['email'])
    if jan_june_rows:
        print(f"Jan/June email fill rate: {jan_june_filled/len(jan_june_rows):.2%}")
    
    # Analyze Safari/Brave/Mobile bias
    safari_brave_mobile = [row for row in data if row['referrer'] in ['safari', 'brave'] or row['device'] == 'mobile']
    sbm_filled = sum(1 for row in safari_brave_mobile if row['email'])
    if safari_brave_mobile:
        print(f"Safari/Brave/Mobile email fill rate: {sbm_filled/len(safari_brave_mobile):.2%}")
    
    # Analyze FATEC bias
    fatec_rows = [row for row in data if '/vestibular/FATEC' in row['route']]
    fatec_filled = sum(1 for row in fatec_rows if row['email'])
    if fatec_rows:
        print(f"FATEC route email fill rate: {fatec_filled/len(fatec_rows):.2%}")

# Main execution
if __name__ == "__main__":
    # Generate synthetic data
    N_ROWS = 1000  # Change this to generate different amounts of data
    
    print(f"Generating {N_ROWS} rows of synthetic web analytics data...")
    synthetic_data = generate_synthetic_data(N_ROWS)
    
    # Print sample
    print_sample_data(synthetic_data)
    
    # Analyze biases
    analyze_biases(synthetic_data)
    
    # Save to CSV
    save_to_csv(synthetic_data, FILENAME)
    
    print(f"\nGeneration complete! {N_ROWS} rows created with specified biases.")