import csv
import random
from datetime import datetime, timedelta
from faker import Faker

# Initialize Faker for Brazilian locale
fake = Faker('pt_BR')

# ==================== PARAMETERS ====================
FILENAME = "datasets/bv-profiles.csv"

NUMBER_OF_ROWS = 1000
MIN_BIRTHDATE = datetime(1980, 1, 1)
MAX_BIRTHDATE = datetime(2008, 12, 31)

# Weight parameters
YOUNG_INCOMPLETE_EDUCATION_BOOST = 0.5  # Added probability for incomplete education if age < 25
HIGHSCHOOL_EXAM_BOOST = 0.25  # Added probability for exams if education is highschool
BASE_PURCHASE_LIKELIHOOD = 0.25
EXTROVERT_COMMUNICATION_BOOST = 0.3  # Boost for communication-related interests
RESEARCHER_UNDERGRAD_EXAM_PENALTY = 0.90  # Reduction in undergrad exam likelihood (5% becomes the result)
RESEARCHER_GRAD_EXAM_BOOST = 0.25  # Boost to graduate exam likelihood (15% -> 50%)
COMPUTING_POSCOMP_BOOST = 0.20
ECONOMICS_ANPEC_BOOST = 0.20

# ==================== DATA DEFINITIONS ====================

# Brazilian states weighted by population and urbanization
STATES = {
    'SP': 30, 'RJ': 15, 'MG': 12, 'BA': 8, 'PR': 7, 'RS': 7,
    'PE': 5, 'CE': 5, 'PA': 4, 'SC': 4, 'GO': 3, 'MA': 2,
    'PB': 2, 'ES': 2, 'PI': 1, 'AL': 1, 'RN': 1, 'MT': 1,
    'MS': 1, 'SE': 1, 'RO': 1, 'TO': 1, 'AC': 1, 'AM': 1,
    'AP': 1, 'RR': 1, 'DF': 3
}

MBTI_TYPES = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP'
]

EDUCATION_LEVELS = ['incomplete', 'highschool', 'undergraduate', 'graduate']

EXPERIENCE_LEVELS = ['unemployed', 'working', 'working on field of study', 'researcher']

# Professional fields (about 50)
INTERESTS = [
    'machine learning', 'data science', 'artificial intelligence', 'software engineering',
    'web development', 'cybersecurity', 'cloud computing', 'database management',
    'music', 'visual arts', 'performing arts', 'graphic design',
    'education', 'psychology', 'sociology', 'philosophy',
    'economics', 'finance', 'accounting', 'business administration',
    'marketing', 'advertising', 'public relations', 'journalism',
    'law', 'political science', 'international relations', 'public policy',
    'medicine', 'nursing', 'pharmacy', 'dentistry',
    'biology', 'chemistry', 'physics', 'mathematics',
    'civil engineering', 'mechanical engineering', 'electrical engineering', 'architecture',
    'environmental science', 'agriculture', 'veterinary medicine', 'nutrition',
    'literature', 'linguistics', 'history', 'anthropology',
    'sports science', 'physical education', 'tourism', 'hospitality'
]

COMMUNICATION_INTERESTS = [
    'marketing', 'advertising', 'public relations', 'journalism',
    'education', 'performing arts', 'public policy', 'business administration'
]

COMPUTING_INTERESTS = [
    'machine learning', 'data science', 'artificial intelligence', 'software engineering',
    'web development', 'cybersecurity', 'cloud computing', 'database management'
]

ECONOMICS_INTERESTS = [
    'economics', 'finance', 'accounting', 'business administration'
]

UNDERGRAD_EXAMS = ['ENEM', 'FUVEST', 'Vestibular UNICAMP', 'Vestibular FATEC']
GRAD_EXAMS = ['POSCOMP', 'ANPEC']

# ==================== HELPER FUNCTIONS ====================

def weighted_choice(choices_dict):
    """Select item based on weights"""
    items = list(choices_dict.keys())
    weights = list(choices_dict.values())
    return random.choices(items, weights=weights, k=1)[0]

def is_extrovert(mbti):
    """Check if MBTI type is extroverted"""
    return mbti[0] == 'E'

def calculate_age(birthdate):
    """Calculate age from birthdate"""
    today = datetime(2025, 9, 29)  # Current date
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

def generate_education(age):
    """Generate education level based on age"""
    if age < 18:
        return 'incomplete'
    elif age < 22:
        # Young people more likely to have incomplete education
        if random.random() < 0.5 + YOUNG_INCOMPLETE_EDUCATION_BOOST:
            return 'incomplete'
        return random.choice(['incomplete', 'highschool', 'undergraduate'])
    elif age < 25:
        if random.random() < 0.3 + YOUNG_INCOMPLETE_EDUCATION_BOOST:
            return 'incomplete'
        return random.choice(['highschool', 'undergraduate'])
    elif age < 30:
        return random.choice(['highschool', 'undergraduate', 'graduate'])
    else:
        # Older people weighted towards higher education
        weights = [5, 20, 50, 25]
        return random.choices(EDUCATION_LEVELS, weights=weights, k=1)[0]

def generate_experience(age, education):
    """Generate experience level based on age and education"""
    if age < 18:
        return 'unemployed'
    elif education == 'incomplete' or education == 'highschool':
        return random.choices(
            EXPERIENCE_LEVELS,
            weights=[30, 60, 5, 5],
            k=1
        )[0]
    elif education == 'undergraduate':
        return random.choices(
            EXPERIENCE_LEVELS,
            weights=[10, 40, 40, 10],
            k=1
        )[0]
    else:  # graduate
        return random.choices(
            EXPERIENCE_LEVELS,
            weights=[5, 45, 30, 20],
            k=1
        )[0]

def generate_interests(mbti):
    """Generate 1-3 interests based on MBTI"""
    num_interests = random.randint(1, 3)
    selected_interests = []
    
    # Extroverts more likely to choose communication fields
    if is_extrovert(mbti):
        if random.random() < EXTROVERT_COMMUNICATION_BOOST:
            comm_interest = random.choice(COMMUNICATION_INTERESTS)
            selected_interests.append(comm_interest)
            num_interests -= 1
    
    # Fill remaining interests
    remaining_interests = [i for i in INTERESTS if i not in selected_interests]
    selected_interests.extend(random.sample(remaining_interests, min(num_interests, len(remaining_interests))))
    
    return '|'.join(selected_interests)

def generate_purchases(education, experience, interests_str, age):
    """Generate purchases based on education, experience, interests, and age"""
    # Base check: 85% of people buy nothing
    if random.random() > BASE_PURCHASE_LIKELIHOOD:
        return ''
    
    # People working on field of study never buy
    if experience == 'working on field of study':
        return ''
    
    purchases = []
    interests_list = interests_str.split('|')
    
    # Determine purchase probabilities
    undergrad_prob = BASE_PURCHASE_LIKELIHOOD
    grad_prob = BASE_PURCHASE_LIKELIHOOD
    
    # High schoolers more likely to buy undergrad exams
    if education == 'highschool':
        undergrad_prob += HIGHSCHOOL_EXAM_BOOST
    
    # Researchers: 5% undergrad, 50% grad
    if experience == 'researcher':
        undergrad_prob = BASE_PURCHASE_LIKELIHOOD * (1 - RESEARCHER_UNDERGRAD_EXAM_PENALTY)
        grad_prob = BASE_PURCHASE_LIKELIHOOD + RESEARCHER_GRAD_EXAM_BOOST
    
    # Interest-based boosts
    has_computing = any(interest in COMPUTING_INTERESTS for interest in interests_list)
    has_economics = any(interest in ECONOMICS_INTERESTS for interest in interests_list)
    
    # Undergraduate exams
    if education in ['incomplete', 'highschool'] or age < 25:
        if random.random() < undergrad_prob:
            num_exams = random.randint(1, 2)
            purchases.extend(random.sample(UNDERGRAD_EXAMS, min(num_exams, len(UNDERGRAD_EXAMS))))
    
    # Graduate exams
    if education in ['undergraduate', 'graduate'] or age >= 22:
        if random.random() < grad_prob:
            # POSCOMP for computing interests
            if has_computing and random.random() < COMPUTING_POSCOMP_BOOST:
                if 'POSCOMP' not in purchases:
                    purchases.append('POSCOMP')
            
            # ANPEC for economics interests
            if has_economics and random.random() < ECONOMICS_ANPEC_BOOST:
                if 'ANPEC' not in purchases:
                    purchases.append('ANPEC')
            
            # Random graduate exam if none selected yet
            if len(purchases) == 0 or (len(purchases) < 2 and random.random() < 0.5):
                available_grad = [e for e in GRAD_EXAMS if e not in purchases]
                if available_grad:
                    purchases.append(random.choice(available_grad))
    
    # Limit to 2 purchases
    return '|'.join(purchases[:2])

# ==================== MAIN GENERATION ====================

def generate_profile_data(num_rows):
    """Generate complete profile dataset"""
    profiles = []
    
    for _ in range(num_rows):
        # Generate birthdate
        time_between_dates = MAX_BIRTHDATE - MIN_BIRTHDATE
        days_between_dates = time_between_dates.days
        random_days = random.randrange(days_between_dates)
        birthdate = MIN_BIRTHDATE + timedelta(days=random_days)
        age = calculate_age(birthdate)
        
        # Generate other fields
        email = fake.email()
        birthdate_str = birthdate.strftime('%d/%m/%Y')
        education = generate_education(age)
        state = weighted_choice(STATES)
        mbti = random.choice(MBTI_TYPES)
        interests = generate_interests(mbti)
        experience = generate_experience(age, education)
        purchases = generate_purchases(education, experience, interests, age)
        
        profiles.append({
            'email': email,
            'birthdate': birthdate_str,
            'education': education,
            'state': state,
            'mbti': mbti,
            'interests': interests,
            'purchases': purchases,
            'experience': experience
        })
    
    return profiles

# ==================== EXPORT TO CSV ====================

def export_to_csv(profiles, filename='profile_data.csv'):
    """Export profiles to CSV file"""
    fieldnames = ['email', 'birthdate', 'education', 'state', 'mbti', 'interests', 'purchases', 'experience']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(profiles)
    
    print(f"✓ Generated {len(profiles)} profiles")
    print(f"✓ Saved to {filename}")

# ==================== RUN ====================

if __name__ == "__main__":
    print("Generating synthetic profile data...")
    profiles = generate_profile_data(NUMBER_OF_ROWS)
    export_to_csv(profiles, FILENAME)
    
    # Print statistics
    print("\n--- Statistics ---")
    with_purchases = sum(1 for p in profiles if p['purchases'])
    print(f"Profiles with purchases: {with_purchases} ({with_purchases/len(profiles)*100:.1f}%)")
    
    education_counts = {}
    for p in profiles:
        education_counts[p['education']] = education_counts.get(p['education'], 0) + 1
    print(f"Education distribution: {education_counts}")
