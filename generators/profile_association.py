import csv
import random
from datetime import datetime
from collections import defaultdict

# ==================== PARAMETERS ====================
PROFILES_CSV = 'datasets/bv-profiles.csv'
NAVIGATION_CSV = 'datasets/bv-web-analytics.csv'
OUTPUT_CSV = 'datasets/bv-web-analytics-associated.csv'

# Weight parameters
TIKTOK_TEENAGER_LIKELIHOOD = 0.75  # 75% chance tiktok referrer gets teenager email
MOBILE_YOUNG_LIKELIHOOD = 0.60     # 60% chance mobile/tablet user is under 20
DESKTOP_OLDER_LIKELIHOOD = 0.60    # 60% chance desktop user is over 24
PURCHASE_ROUTE_MATCH_BOOST = 0.70  # 70% boost to spend time on purchased exam route

# ==================== HELPER FUNCTIONS ====================

def calculate_age(birthdate_str):
    """Calculate age from birthdate string (dd/mm/YYYY)"""
    birthdate = datetime.strptime(birthdate_str, '%d/%m/%Y')
    today = datetime(2025, 9, 29)
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

def load_profiles(filename):
    """Load profiles and organize by age groups and purchases"""
    profiles = []
    profiles_by_age = defaultdict(list)
    profiles_with_purchases = defaultdict(list)  # exam -> list of profiles
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            age = calculate_age(row['birthdate'])
            row['age'] = age
            profiles.append(row)
            
            # Group by age ranges
            if age < 20:
                profiles_by_age['teenager'].append(row)
            elif age < 25:
                profiles_by_age['young_adult'].append(row)
            else:
                profiles_by_age['adult'].append(row)
            
            # Group by purchases
            if row['purchases']:
                purchases = row['purchases'].split('|')
                for purchase in purchases:
                    profiles_with_purchases[purchase.strip()].append(row)
    
    return profiles, profiles_by_age, profiles_with_purchases

def extract_exam_from_route(route):
    """Extract exam name from route like /vestibular/ENEM"""
    if route.startswith('/vestibular/'):
        parts = route.split('/')
        if len(parts) >= 3:
            return parts[2]
    return None

def select_profile(profiles, profiles_by_age, profiles_with_purchases, nav_row):
    """Select appropriate profile based on navigation data and weights"""
    referrer = nav_row.get('referrer', '')
    device = nav_row.get('device', '')
    route = nav_row.get('route', '')
    
    # Start with all profiles as candidates
    candidates = profiles.copy()
    weights_applied = []
    
    # Weight 1: TikTok referrer -> teenager (under 20)
    if referrer.lower() == 'tiktok':
        if random.random() < TIKTOK_TEENAGER_LIKELIHOOD:
            if profiles_by_age['teenager']:
                candidates = profiles_by_age['teenager']
                weights_applied.append('tiktok_teenager')
    
    # Weight 2: Mobile/Tablet -> likely under 20 (60%)
    elif device.lower() in ['mobile', 'tablet']:
        if random.random() < MOBILE_YOUNG_LIKELIHOOD:
            if profiles_by_age['teenager']:
                candidates = profiles_by_age['teenager']
                weights_applied.append('mobile_young')
    
    # Weight 3: Desktop -> likely over 24 (60%)
    elif device.lower() == 'desktop':
        if random.random() < DESKTOP_OLDER_LIKELIHOOD:
            if profiles_by_age['adult']:
                candidates = profiles_by_age['adult']
                weights_applied.append('desktop_adult')
    
    # Weight 4: Route matches purchase -> prefer users with that purchase
    exam = extract_exam_from_route(route)
    if exam and exam in profiles_with_purchases:
        if random.random() < PURCHASE_ROUTE_MATCH_BOOST:
            # Try to find candidate with this purchase
            matching_profiles = [p for p in candidates if exam in p.get('purchases', '')]
            if matching_profiles:
                candidates = matching_profiles
                weights_applied.append(f'purchase_match_{exam}')
            elif profiles_with_purchases[exam]:
                # If no overlap in candidates, use all profiles with this purchase
                candidates = profiles_with_purchases[exam]
                weights_applied.append(f'purchase_match_{exam}_override')
    
    # Select random profile from candidates
    if candidates:
        selected = random.choice(candidates)
        return selected['email'], weights_applied
    
    # Fallback to random profile
    return random.choice(profiles)['email'], ['fallback']

def process_navigation_data(profiles_csv, navigation_csv, output_csv):
    """Read navigation data and replace emails with matching profiles"""
    print(f"Loading profiles from {profiles_csv}...")
    profiles, profiles_by_age, profiles_with_purchases = load_profiles(profiles_csv)
    
    print(f"Profiles loaded: {len(profiles)}")
    print(f"  - Teenagers (<20): {len(profiles_by_age['teenager'])}")
    print(f"  - Young adults (20-24): {len(profiles_by_age['young_adult'])}")
    print(f"  - Adults (25+): {len(profiles_by_age['adult'])}")
    print(f"  - With purchases: {sum(len(v) for v in profiles_with_purchases.values())}")
    
    print(f"\nProcessing navigation data from {navigation_csv}...")
    
    # Read navigation data
    rows_processed = 0
    emails_replaced = 0
    anonymous_rows = 0
    weight_stats = defaultdict(int)
    
    with open(navigation_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        # Find email column index
        if fieldnames is not None and 'email' not in fieldnames:
            print("ERROR: 'email' column not found in navigation CSV!")
            return
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames or [])
        writer.writeheader()
        
        for row in reader:
            rows_processed += 1
            
            # Check if row has email (non-empty and not just whitespace)
            if row['email'] and row['email'].strip():
                # Replace with matching profile email
                new_email, weights = select_profile(profiles, profiles_by_age, profiles_with_purchases, row)
                row['email'] = new_email
                emails_replaced += 1
                
                # Track weight statistics
                for weight in weights:
                    weight_stats[weight] += 1
            else:
                # Keep as anonymous (empty)
                row['email'] = ''
                anonymous_rows += 1
            
            writer.writerow(row)
    
    # Print statistics
    print(f"\n✓ Processing complete!")
    print(f"  - Total rows processed: {rows_processed}")
    print(f"  - Emails replaced: {emails_replaced}")
    print(f"  - Anonymous rows kept: {anonymous_rows}")
    print(f"  - Output saved to: {output_csv}")
    
    if weight_stats:
        print(f"\n--- Weight Application Statistics ---")
        for weight, count in sorted(weight_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / emails_replaced * 100) if emails_replaced > 0 else 0
            print(f"  - {weight}: {count} ({percentage:.1f}%)")

# ==================== MAIN ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Navigation Data Email Replacement Script")
    print("=" * 60)
    
    try:
        process_navigation_data(PROFILES_CSV, NAVIGATION_CSV, OUTPUT_CSV)
        print("\n✓ Script completed successfully!")
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: File not found - {e}")
        print("  Make sure both CSV files exist in the same directory.")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()